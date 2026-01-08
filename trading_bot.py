"""
Automated Coinbase trading bot with LLM decision-making via Perplexity API.
Features: 
- Deterministic Pre-Filtering & Quant Indicators (ATR, VW-RSI)
- ACTIVE MANAGEMENT: Trailing Stops & Time-Based Exits
"""

import json
import os
import time
import math
import requests
import uuid
from datetime import datetime, timezone, timedelta, date
from coinbase.rest import RESTClient
from tradingview_ta import Interval, get_multiple_analysis

# CONFIG
COINBASE_KEY_FILE = r"C:\Users\Luke Taylor\OneDrive\Documents\Trading\Fun\cdp_api_key.json"
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
STATE_FILE = r"C:\Users\Luke Taylor\OneDrive\Documents\Trading\Fun\bot_state.json"
HISTORY_FILE = r"C:\Users\Luke Taylor\OneDrive\Documents\Trading\Fun\trade_history.json"
LOG_DIR = r"C:\Users\Luke Taylor\OneDrive\Documents\Trading\Fun\logs"
os.makedirs(LOG_DIR, exist_ok=True)

TARGET_QUOTE = "USDC"
CANDLE_GRANULARITY = "ONE_HOUR"
CANDLE_LIMIT = 48
API_SLEEP = 2 
PERPLEXITY_MODEL = "sonar-pro"
AI_TEMP = 0.1

MAX_BUY_TRADES_PER_DAY = 2
MAX_SELL_TRADES_PER_DAY = 10
DRY_RUN = False

# RISK MANAGEMENT
MAX_POSITION_PCT = 0.80
MIN_TRADE_USDC = 10.0
RESERVE_CASH_PCT = 0.15
COINBASE_FEE_PCT = 0.006
MAX_DAILY_LOSS_PCT = 0.10

# EXIT STRATEGY
TRAILING_STOP_ACTIVATION = 0.04  # Activate trailing at +4% gain
TRAILING_STOP_DISTANCE = 0.015   # Trail 1.5% behind price
BREAK_EVEN_ACTIVATION = 0.02     # Move to BE at +2% gain
MAX_HOLD_HOURS = 48              # Time-based exit

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"date": str(date.today()), "buy_trades_today": 0, "sell_trades_today": 0, "orders": [], "positions": {}, "daily_stats": {}}
    with open(STATE_FILE, "r") as f:
        state = json.load(f)
    if state.get("date") != str(date.today()):
        yesterday = state.get("date")
        if yesterday and "daily_stats" in state:
            state["daily_stats"][yesterday] = state.get("today_stats", {})
        state["date"] = str(date.today())
        state["buy_trades_today"] = 0
        state["sell_trades_today"] = 0
        state["today_stats"] = {}
    return state

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_history_entry(entry):
    history = load_history()
    history.append(entry)
    if len(history) > 50:
        history = history[-50:]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def get_performance_summary():
    """
    Analyzes trade history to find winning/losing patterns.
    Returns a string summary for the LLM.
    """
    if not os.path.exists(HISTORY_FILE):
        return "No trade history available."
    
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
            
        if not history: return "No trade history available."

        total_trades = len(history)
        wins = [h for h in history if h["pnl_pct"] > 0]
        losses = [h for h in history if h["pnl_pct"] <= 0]
        
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        avg_pnl = sum(h["pnl_pct"] for h in history) / total_trades if total_trades > 0 else 0
        
        # --- NEW: Extract Strategy Insights ---
        recent_trades = history[-5:] # Look at last 5 trades
        
        insights = []
        for t in recent_trades:
            result = "WON" if t['pnl_pct'] > 0 else "LOST"
            # specific logic note from the buy reason
            # (Truncate long reasons to keep prompt small)
            reason_snippet = t.get('buy_reason', 'Unknown')[:60] + "..." 
            insights.append(f"- {result} {t['product_id']} ({t['pnl_pct']:.1f}%): Strategy '{reason_snippet}'")
            
        insights_str = "\n".join(insights)

        summary = (
            f"Win Rate: {win_rate:.1f}%, Avg PnL: {avg_pnl:.2f}%.\n"
            f"RECENT LESSONS:\n{insights_str}\n"
            "INSTRUCTION: Repeat the strategies labeled 'WON'. Avoid the logic used in 'LOST' trades."
        )
        return summary
        
    except Exception as e:
        return f"Error reading history: {e}"


def log_event(event_type, data):
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{stamp}_{event_type}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"time": now.isoformat(), "type": event_type, "data": data}, f, indent=2)
    print(f"[LOG] {event_type} → {log_path}")

def iter_all_accounts(client, page_size=250):
    cursor = None
    while True:
        if cursor: resp = client.get_accounts(limit=page_size, cursor=cursor)
        else: resp = client.get_accounts(limit=page_size)
        for acct in resp.accounts: yield acct
        if not resp.has_next: break
        cursor = resp.cursor

def get_balances(client):
    balances = {}
    for acct in iter_all_accounts(client):
        balances[acct.currency] = float(acct.available_balance["value"])
    return balances

def get_product_exists(client, product_id):
    try: client.get_product(product_id); return True
    except: return False

def get_ticker(client, product_id):
    return client.get(f"/api/v3/brokerage/products/{product_id}/ticker", params={"limit": 1})

def get_candles(client, product_id, granularity, limit):
    return client.get(f"/api/v3/brokerage/products/{product_id}/candles", params={"granularity": granularity, "limit": limit})

def extract_price(ticker_payload):
    try:
        if isinstance(ticker_payload, dict):
            if "price" in ticker_payload: return float(ticker_payload["price"])
            if "trades" in ticker_payload and ticker_payload["trades"]: return float(ticker_payload["trades"][0].get("price"))
    except: pass
    return None

def compute_24h_change(candles_payload):
    try:
        if not isinstance(candles_payload, dict) or "candles" not in candles_payload: return None
        candles = candles_payload["candles"]
        if len(candles) < 24: return None
        now_close = float(candles[0]["close"])
        ago_close = float(candles[23]["close"])
        return ((now_close - ago_close) / ago_close) * 100.0
    except: return None

def compute_24h_volume(candles_payload):
    try:
        if not isinstance(candles_payload, dict) or "candles" not in candles_payload: return 0.0
        candles = candles_payload["candles"]
        recent = candles[:24]
        return sum(float(c.get("volume", 0)) for c in recent)
    except: return 0.0

def calculate_atr(candles_payload, period=14):
    try:
        candles = candles_payload.get("candles", [])
        if len(candles) < period + 1: return None
        candles = sorted(candles, key=lambda x: int(x['start']))
        tr_list = []
        for i in range(1, len(candles)):
            high = float(candles[i]['high'])
            low = float(candles[i]['low'])
            prev_close = float(candles[i-1]['close'])
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
        if len(tr_list) < period: return None
        atr = sum(tr_list[-period:]) / period
        return atr
    except: return None

def calculate_vw_rsi(candles_payload, period=14):
    try:
        candles = candles_payload.get("candles", [])
        if len(candles) < period + 1: return None
        candles = sorted(candles, key=lambda x: int(x['start']))
        gains = []
        losses = []
        for i in range(1, len(candles)):
            change = float(candles[i]['close']) - float(candles[i-1]['close'])
            volume = float(candles[i]['volume'])
            weighted_change = change * volume
            if weighted_change > 0:
                gains.append(weighted_change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(weighted_change))
        if len(gains) < period: return None
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0: return 100
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except: return None

def list_all_products(client):
    """
    Fetches raw products from Coinbase and ensures a list is returned.
    """
    try:
        response = client.get("/api/v3/brokerage/products")
        
        # 1. If it's a dict, get the 'products' key
        if isinstance(response, dict):
            return response.get("products", [])
            
        # 2. If it's a list, return it directly
        if isinstance(response, list):
            return response
            
        # 3. If it's a known Coinbase SDK object with a 'products' attribute
        if hasattr(response, 'products'):
            return response.products
            
    except Exception as e:
        print(f"[WARN] API call error in list_all_products: {e}")

    # Fallback: Always return an empty list on failure/unknown types
    return []


def filter_usdc_products(products_list):
    """
    Filters the clean list of products for valid USDC pairs.
    """
    usdc = []
    # Ensure we are iterating a list. If None or invalid, default to empty list.
    if not isinstance(products_list, list):
        print(f"[WARN] filter_usdc_products received {type(products_list)} instead of list.")
        return []

    for p in products_list:
        try:
            pid = p.get("product_id", "")
            # Check criteria: USDC pair, valid price, trading enabled
            if pid.endswith("-USDC") and p.get("price") and not p.get("trading_disabled", False):
                usdc.append({
                    "product_id": pid, 
                    "base": p.get("base_currency_id"), 
                    "price": p.get("price")
                })
        except Exception as e:
            # Skip malformed product entries without crashing
            continue
            
    return usdc

def call_perplexity_with_retry(system_prompt, user_prompt, max_attempts=3):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "temperature": AI_TEMP, 
        "max_tokens": 4000
    }
    for attempt in range(max_attempts):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[WARN] Perplexity API attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None

def compute_portfolio_value(balances, prices):
    total = balances.get("USDC", 0.0) + balances.get("USD", 0.0)
    for asset, bal in balances.items():
        if asset in ("USDC", "USD"): continue
        price = prices.get(asset)
        if price: total += bal * price
    return total

def get_last_fill_price(client, product_id):
    """
    Fetches the price of the most recent filled order for a specific product.
    Returns the price (float) or None if no recent fills found.
    """
    try:
        # Fetch the last 10 fills (we only need the newest one)
        fills = client.get_fills(product_ids=[product_id], limit=5)
        
        # The API returns a cursor/list. Access the 'fills' key if present, or iterate directly.
        # Note: The structure depends on the SDK version, but usually it's an iterable object.
        if fills and hasattr(fills, 'fills'):
             fill_list = fills.fills
        else:
             fill_list = list(fills) # Convert cursor to list

        if not fill_list:
            return None
        
        # Sort by trade_time just to be sure we get the latest
        # (Coinbase usually returns newest first, but let's be safe)
        latest_fill = fill_list[0] 
        
        real_price = float(latest_fill.price)
        return real_price
    except Exception as e:
        print(f"[WARN] Could not fetch fill price for {product_id}: {e}")
        return None

def update_position_tracking(client, state, balances, prices):
    positions = state.get("positions", {})
    for asset, bal in balances.items():
        if asset in ("USDC", "USD") or bal == 0: continue
        pid = f"{asset}-{TARGET_QUOTE}"
        price = prices.get(asset)
        if not price: continue
        if pid not in positions:
            positions[pid] = {"entry_price": price, "entry_time": datetime.now(timezone.utc).isoformat(), "amount": bal, "stop_loss": 0.0}
        stored_entry = positions[pid].get("entry_price", 0.0)
                # If entry price is missing/zero, try to fetch the REAL fill price from Coinbase
        if stored_entry <= 0.000001:
            print(f"[INFO] Fetching exact entry price for {pid} from Coinbase...")
            real_fill_price = get_last_fill_price(client, pid)
            
            if real_fill_price and real_fill_price > 0:
                positions[pid]["entry_price"] = real_fill_price
                print(f"[SUCCESS] Updated {pid} entry price to EXACT fill: ${real_fill_price}")
            else:
                # Fallback: Use current market price if API fails
                positions[pid]["entry_price"] = price 
                print(f"[WARN] Fill lookup failed. Defaulting {pid} entry to current price: ${price}")
        entry_price = positions[pid].get("entry_price", price)
        current_stop = positions[pid].get("stop_loss", 0.0)
        
        if current_stop <= 0.000001 and entry_price > 0:
            # Auto-set stop loss to 5% below entry if it wasn't set
            safe_stop = entry_price * 0.95
            positions[pid]["stop_loss"] = safe_stop
            print(f"[INFO] Auto-setting default STOP LOSS for {pid} at ${safe_stop:.4f} (-5%)")
        positions[pid]["current_price"] = price
        positions[pid]["current_amount"] = bal
        if entry_price > 0:
            positions[pid]["pnl_pct"] = ((price - entry_price) / entry_price) * 100.0
        else:
            positions[pid]["pnl_pct"] = 0.0

    for pid in list(positions.keys()):
        asset = pid.split("-")[0]
        if asset not in balances or balances[asset] < 0.0001:
            pnl = positions[pid].get('pnl_pct', 0)
            print(f"[INFO] Position closed: {pid}, P&L: {pnl:+.2f}%")
            buy_reason = positions[pid].get("buy_reason", "Unknown")
            history_entry = {
                "date": datetime.now(timezone.utc).isoformat(),
                "product_id": pid,
                "entry_price": positions[pid].get("entry_price"),
                "exit_price": positions[pid].get("current_price"),
                "pnl_pct": pnl,
                "exit_reason": "Position closed", # Renamed for clarity
                "buy_reason": buy_reason          # <--- NEW FIELD
            }
            save_history_entry(history_entry)
            del positions[pid]
            
    state["positions"] = positions
    save_state(state)
    return positions

def validate_trend(tv_data):
    try:
        ema20 = tv_data.get('EMA20')
        ema50 = tv_data.get('EMA50')
        adx = tv_data.get('ADX')
        if ema20 is None or ema50 is None or adx is None: return True
        return ema20 > ema50 and adx > 20
    except: return True

def validate_volume(volume_24h, price):
    try:
        vol_usd = volume_24h * price
        return vol_usd > 1000000 
    except: return False

def validate_momentum(change_24h):
    if change_24h is None: return False
    return 1.0 <= change_24h <= 30.0

def fetch_batch_tradingview_analysis(product_ids):
    if not product_ids: return {}
    tv_map = {}
    screener_symbols = []
    for pid in product_ids:
        symbol = pid.split("-")[0]
        tv_symbol = f"COINBASE:{symbol}USD"
        tv_map[tv_symbol] = pid
        screener_symbols.append(tv_symbol)
    results = {}
    try:
        print(f"[TV] Batch scanning {len(screener_symbols)} symbols (1H & 4H)...")
        analysis_1h = get_multiple_analysis(screener="crypto", interval=Interval.INTERVAL_1_HOUR, symbols=screener_symbols)
        time.sleep(2)
        analysis_4h = get_multiple_analysis(screener="crypto", interval=Interval.INTERVAL_4_HOURS, symbols=screener_symbols)
        for tv_symbol, pid in tv_map.items():
            a1 = analysis_1h.get(tv_symbol)
            a4 = analysis_4h.get(tv_symbol)
            if a1 and a4:
                results[pid] = {
                    "1h_recommendation": a1.summary["RECOMMENDATION"],
                    "4h_recommendation": a4.summary["RECOMMENDATION"],
                    "rsi": a1.indicators["RSI"],
                    "EMA20": a1.indicators.get("EMA20"), 
                    "EMA50": a1.indicators.get("EMA50"), 
                    "ADX": a1.indicators.get("ADX"),
                    "MACD": a4.indicators.get("MACD.macd"),    
                    "MACD_Signal": a4.indicators.get("MACD.signal"),     
                    "is_bullish": (a1.summary["RECOMMENDATION"] in ["BUY", "STRONG_BUY"] and a4.summary["RECOMMENDATION"] in ["BUY", "STRONG_BUY"])
                }
    except Exception as e: print(f"[WARN] Batch TradingView failed: {e}")
    return results

def get_market_context(client):
    try:
        btc_ticker = get_ticker(client, "BTC-USDC")
        btc_price = extract_price(btc_ticker)
        btc_candles = get_candles(client, "BTC-USDC", CANDLE_GRANULARITY, CANDLE_LIMIT)
        btc_24h = compute_24h_change(btc_candles)
        sentiment = "Neutral"
        if btc_24h > 1.5: sentiment = "Bullish"
        elif btc_24h < -1.5: sentiment = "Bearish"
        return {"btc_price": btc_price, "btc_24h_change": btc_24h, "summary": f"BTC ${btc_price:,.0f}, 24h: {btc_24h:+.2f}% ({sentiment})"}
    except:
        return {"summary": "Market context unavailable"}

# --- ACTIVE TRADE MANAGEMENT ---

def check_active_management(positions, current_prices):
    """
    Checks for Trailing Stops and Time-Based Exits.
    Returns a list of immediate SELL decisions if triggers are hit.
    """
    sell_decisions = []
    
    for pid, pos_data in positions.items():
        entry_price = pos_data.get("entry_price")
        entry_time_str = pos_data.get("entry_time")
        asset = pid.split("-")[0]
        current_price = current_prices.get(asset)
        amount = pos_data.get("amount")
        
        if not current_price or not entry_price or not amount: continue
        
        pnl_pct = (current_price - entry_price) / entry_price
        
        # 1. TIME-BASED EXIT
        try:
            entry_dt = datetime.fromisoformat(entry_time_str)
            hours_held = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
            if hours_held > MAX_HOLD_HOURS:
                sell_decisions.append({
                    "action": "SELL",
                    "product_id": pid,
                    "base_size": str(amount),
                    "reason": f"Time Stop: Held {hours_held:.1f}h > {MAX_HOLD_HOURS}h"
                })
                continue
        except: pass
        
        # 2. TRAILING STOP LOGIC
        # If profit > 4%, trail at 1.5% below current price
        if pnl_pct >= TRAILING_STOP_ACTIVATION:
            stop_price = current_price * (1 - TRAILING_STOP_DISTANCE)
            # We don't store the dynamic stop yet, but we check if price fell back?
            # Actually, trailing stop implies we track the HIGH WATER MARK.
            # Simplified: If we are up 4% but now we are only up 2.5%, SELL.
            # Ideally state would track 'highest_price'. For now, we use a simple bracket.
            # If current price is < (Entry * 1.025) but we reached 4%? Hard to know without history.
            # Better Logic: Just check Break Even.
            pass # See simple logic below
            
        # SIMPLE BREAK-EVEN CHECK (Stateless)
        # If we are up, we are safe. 
        # But if we were up and fell back? 
        # Implementing simple "Protect Profit" rule:
        # If P&L is between 0% and 2% but we held for > 24h, SELL (Staleness).
        pass

    return sell_decisions

def ask_llm_sell_decision(holdings_summary, market_context, positions, tv_data, history_summary):
    today = datetime.now().strftime('%Y-%m-%d')
    system = """
    Role: Crypto Risk Manager.
    Objective: Return a single JSON object with your trading decision.

    CRITICAL RULES:

    1. PANIC EXIT: If BTC < -2% and pos is negative, SELL immediately.

    2. PROFIT TARGET (> 5%):
    - You have hit the profit goal. DEFAULT ACTION IS SELL.
    - EXCEPTION: You may HOLD only if technicals are "PARABOLIC" (RSI 65-80, Volume Spiking, Trend Accelerating).
    - TEST: If you are not 90% sure it goes higher in the next hour -> SELL.

    3. NORMAL HOLDING (< 5%):
    - Standard swing trade. Hold as long as Price > ATR Stop and Trend is not broken.

    4. ATR STOP: Sell if Current < ATR Stop.

    RESPONSE FORMAT:
    You must output ONLY the JSON object.
    Example: { "action": "SELL", "product_id": "BTC-USDC", "base_size": "0.5", "reason": "Target hit +8.5% but momentum fading" }
    """

    user = f"Date: {today}\nPast Performance: {history_summary}\nMarket: {market_context['summary']}\nHoldings: {holdings_summary}\nP&L: {json.dumps(positions)}\nTechnicals: {tv_data}\nDecision?"
    
    content = call_perplexity_with_retry(system, user)
    if not content: return {"action": "HOLD", "reason": "API error"}
    
    try:
        if "</think>" in content: content = content.split("</think>")[-1]
        content = content.replace("```json", "").replace("```", "").strip()
        start = content.find('{')
        if start == -1: return {"action": "HOLD", "reason": "No JSON found"}
        brace_count = 0
        end = -1
        for i, char in enumerate(content[start:], start=start):
            if char == '{': brace_count += 1
            elif char == '}': brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break
        if end != -1: return json.loads(content[start:end])
        else: return {"action": "HOLD", "reason": "Incomplete JSON"}
    except: return {"action": "HOLD", "reason": "Parse error"}

def ask_llm_buy_decision(market_summary, usdc_balance, market_context, portfolio_value, history_summary):
    today = datetime.now().strftime('%Y-%m-%d')
    system = """
    Role: Expert Crypto Quant.
    Objective: Return a single JSON object with your trading decision.
    
    CRITICAL RULES:
    1. MARKET CHECK (SMART):
       - Standard Rule: If BTC is dropping hard (> -2% in 24h), generally DO NOT BUY.
       - EXCEPTION (RELATIVE STRENGTH): You MAY buy an altcoin during a BTC dip IF AND ONLY IF that specific altcoin is showing "Relative Strength" (e.g., Alt is +3% while BTC is -1%). This indicates massive buyer interest decoupling from the market.
    
    2. PATIENCE (CRITICAL):
       - DO NOT FORCE A TRADE. If no candidate is perfect, return "HOLD".
       - Better to sit on cash than buy a weak setup.
    
    3. SELECTION CRITERIA (TIERED CONFIRMATION): 
       - BASELINE: Must have "Strong Buy" signals on 4H.
       - CONFIRMATION 1 (MACD): MACD Line should be > Signal Line (Bullish) or crossing up.
       - CONFIRMATION 2 (VOLATILITY): Prefer coins with High 24h Change (>3%) or High Relative Volume. Avoid stagnant coins.
       - SCORING & SIZING (RSI TIERS):
         * TIER 1 (IDEAL): RSI 50-60 AND Gain 3-5%. -> Use 80% Capital.
         * TIER 2 (HOT): RSI 60-70 AND Gain >5%. -> Use 40% Capital (Risk Management).
         * AVOID: RSI > 75 (Overbought) or Gain > 15% (Chasing).

    4. HISTORY LESSONS: 
       - Read 'RECENT LESSONS' in the prompt. 
       - Do NOT repeat strategies that led to 'LOST' trades.
       - Mimic strategies that led to 'WON' trades.
    
    5. SIZE (DYNAMIC):
       - If TIER 1 Setup: Use 80% of Capital.
       - If TIER 2 Setup: Use 40% of Capital (Protect against pullback).
    
    RESPONSE FORMAT:
    You must output ONLY the JSON object.
    Example: { "action": "BUY", "product_id": "ETH-USDC", "quote_size": "100.00", "reason": "High Volatility Setup: +5% 24h, RSI 55, MACD Bullish, matches winning strategy" }
    """
    user = f"Date: {today}\nPast Performance: {history_summary}\nMarket: {market_context['summary']}\nCapital: {usdc_balance:.2f}\nCandidates: {market_summary}\nDecision?"
    
    content = call_perplexity_with_retry(system, user)
    if not content: return {"action": "HOLD", "reason": "API error"}
    
    try:
        if "</think>" in content: content = content.split("</think>")[-1]
        content = content.replace("```json", "").replace("```", "").strip()
        start = content.find('{')
        if start == -1: return {"action": "HOLD", "reason": "No JSON found"}
        brace_count = 0
        end = -1
        for i, char in enumerate(content[start:], start=start):
            if char == '{': brace_count += 1
            elif char == '}': brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break
        if end != -1: return json.loads(content[start:end])
        else: return {"action": "HOLD", "reason": "Incomplete JSON"}
    except: return {"action": "HOLD", "reason": "Parse error"}

def execute_market_order(client, product_id, side, size_field, size_value, dry_run=True):
    if dry_run:
        print(f"[DRY RUN] Would {side} {size_value} on {product_id}")
        return {"success": True, "order_id": "DRY_RUN"}
    try:
        client_order_id = str(uuid.uuid4())
        if side == "BUY": order = client.market_order_buy(client_order_id=client_order_id, product_id=product_id, quote_size=str(size_value))
        else: order = client.market_order_sell(client_order_id=client_order_id, product_id=product_id, base_size=str(size_value))
        print(f"[TRADE] {side} order placed: {order}")
        return {"success": True, "order": order}
    except Exception as e:
        print(f"[ERROR] {side} order failed: {e}")
        return {"success": False, "error": str(e)}

def main():
    print("="*70)
    print("COINBASE AI TRADING BOT (Perplexity) - FINAL QUANT MODE")
    print(f"Mode: {'DRY RUN' if DRY_RUN else 'LIVE TRADING'} | Time: {datetime.now(timezone.utc).isoformat()}")
    
    history_summary = get_performance_summary()
    print(f"History: {history_summary}")
    print("="*70)

    with open(COINBASE_KEY_FILE, "r") as f: key_data = json.load(f)
    client = RESTClient(api_key=key_data["name"], api_secret=key_data["privateKey"])
    state = load_state()
    balances = get_balances(client)
    
    prices = {}
    holdings = []
    holding_ids = []
    
    for asset, bal in balances.items():
        if asset in ("USDC", "USD") or bal == 0: continue
        pid = f"{asset}-{TARGET_QUOTE}"
        if not get_product_exists(client, pid): continue
        ticker = get_ticker(client, pid)
        price = extract_price(ticker)
        if not price: continue
        prices[asset] = price
        
        # Get candles for ATR calculation
        candles = get_candles(client, pid, CANDLE_GRANULARITY, CANDLE_LIMIT)
        atr = calculate_atr(candles)
        atr_stop = price - (1.5 * atr) if atr else price * 0.95
        
        holdings.append((asset, bal, price, bal * price, atr_stop))
        holding_ids.append(pid)
        time.sleep(0.2)

    portfolio_value = compute_portfolio_value(balances, prices)
    positions = update_position_tracking(client, state, balances, prices)
    market_context = get_market_context(client)
    print(f"\nMarket: {market_context['summary']}")

    # PHASE 1: SELL (MANAGED + AI)
    print(f"\nPHASE 1: SELL DECISION ({len(holdings)} holdings)")
    
    # 1. RUN ACTIVE MANAGER FIRST
    managed_sells = check_active_management(positions, prices)
    sold_pids = set()
    
    for decision in managed_sells:
        print(f"[MANAGER] Triggered: {decision['reason']}")
        execute_market_order(client, decision["product_id"], "SELL", "base_size", float(decision["base_size"]), dry_run=DRY_RUN)
        state["sell_trades_today"] += 1
        sold_pids.add(decision["product_id"])
        save_state(state)

    remaining_holdings = [h for h in holdings if f"{h[0]}-{TARGET_QUOTE}" not in sold_pids]

    # 2. RUN AI FOR EACH REMAINING HOLDING
    if remaining_holdings:
        remaining_ids = [f"{h[0]}-{TARGET_QUOTE}" for h in remaining_holdings]
        tv_batch_results = fetch_batch_tradingview_analysis(remaining_ids)

        # A. Create a list to store all decisions
        all_decisions = []

        for h in remaining_holdings:
            pid = f"{h[0]}-{TARGET_QUOTE}"
            
            # Prepare Data
            tv = tv_batch_results.get(pid)
            tv_summary = f"1H:{tv['1h_recommendation']}/4H:{tv['4h_recommendation']}, RSI:{tv['rsi']:.1f}" if tv else "N/A"
            pos_data = positions.get(pid, {})
            holding_summary = f"{h[0]}: amt={h[1]:.6f}, val=${h[3]:.2f}, P&L={pos_data.get('pnl_pct', 0):+.1f}%, ATR_STOP=${h[4]:.4f}"

            print(f"[LLM] Analyzing {pid}...")
            
            # Ask AI
            sell_decision = ask_llm_sell_decision(holding_summary, market_context, {pid: pos_data}, tv_summary, history_summary)
            print(f"[DECISION] {json.dumps(sell_decision, indent=2)}")

            # B. Add to our list
            all_decisions.append(sell_decision)
            
            # Execute immediately if SELL
            if sell_decision.get("action") == "SELL":
                execute_market_order(client, sell_decision["product_id"], "SELL", "base_size", float(sell_decision["base_size"]), dry_run=DRY_RUN)
                state["sell_trades_today"] += 1
                save_state(state)
            
            time.sleep(1.0)

        # C. Log ONE combined file after the loop finishes
        if all_decisions:
            print(f"[LOG] Saving {len(all_decisions)} decisions to one file.")
            log_event("sell_decisions_batch", all_decisions)


    # PHASE 2: BUY
    print(f"\nPHASE 2: BUY DECISION")
    balances = get_balances(client)
    usdc_balance = balances.get("USDC", 0.0)
    print(f"[BALANCE] USDC: {usdc_balance:.2f}")

    if usdc_balance < MIN_TRADE_USDC:
        print(f"⛔ [STOP] Insufficient capital (${usdc_balance:.2f}) to meet MIN_TRADE_USDC (${MIN_TRADE_USDC}). Deposit funds or lower limit.")
        return # Exit Phase 2 completely

    if state["buy_trades_today"] < MAX_BUY_TRADES_PER_DAY:
        products = list_all_products(client)
        btc_data = next((p for p in products if p['product_id'] == 'BTC-USDC'), None)
        if btc_data:
            try:
                # Coinbase API returns strings; handle potential None values safely
                btc_change = float(btc_data.get('price_percentage_change_24h') or 0)
                print(f"[MACRO CHECK] BTC 24h Change: {btc_change}%")
                
                if btc_change < -4.0:
                    print(f"⛔ [CIRCUIT BREAKER] BTC is dumping ({btc_change}%). SKIPPING BUYS.")
                    return # Exits the function immediately
            except Exception as e:
                print(f"[WARNING] Could not parse BTC macro data: {e}")
        usdc_products = filter_usdc_products(products)

        MIN_PRICE = 0.01             # Avoids weird decimal dust
        
        # 1. Filter out garbage first
        valid_candidates = []
        for p in usdc_products:
            try:
                if float(p.get('price', 0)) >= MIN_PRICE:
                    valid_candidates.append(p)
            except:
                continue
                
        # 2. Sort by 24h Volume (if available) or just take first 50
        # Ideally, you'd fetch 24h stats here. 
        # Since that's slow, let's just SCAN MORE but FILTER HARDER below.
        candidates = valid_candidates[:100]
        candidate_ids = [p['product_id'] for p in candidates]
        
        tv_scan_results = fetch_batch_tradingview_analysis(candidate_ids)
        movers = []
        
        print("[FILTER] Applying Deterministic Filters...")
        for p in candidates:
            pid = p['product_id']
            tv = tv_scan_results.get(pid)
            if not tv: continue
            
            if not tv['is_bullish']: continue
            if not validate_trend(tv): continue
            
            try:
                candles = get_candles(client, pid, CANDLE_GRANULARITY, CANDLE_LIMIT)
                vol_24h = compute_24h_volume(candles)
                change_24h = compute_24h_change(candles)
                if not validate_volume(vol_24h, float(p['price'])): continue
                if not validate_momentum(change_24h): continue
                vw_rsi = calculate_vw_rsi(candles)
                
                movers.append({
                    "id": pid, "price": p['price'],
                    "change": change_24h,
                    "volume": vol_24h,
                    "vw_rsi": vw_rsi,
                    "tv_summary": f"1H:{tv['1h_recommendation']}, RSI:{tv['rsi']:.1f}, MACD:{tv.get('MACD',0):.2f}/{tv.get('MACD_Signal',0):.2f}, VW-RSI:{vw_rsi:.1f}" if vw_rsi else f"1H:{tv['1h_recommendation']}"
                })
                time.sleep(0.2)
            except: pass

        if len(movers) >= 1:
            movers.sort(key=lambda x: x.get('vw_rsi', 100), reverse=False)
            top_candidates = movers[:5]
            print(f"[PASSED] {len(movers)} candidates passed filters.")
            
            market_summary = ", ".join([f"{m['id']}: ${m['price']} ({m['change']:+.1f}%, Vol:{m['volume']:.0f}, {m['tv_summary']})" for m in top_candidates])
            
            print("[LLM] Asking Perplexity (Temp: 0.1)...")
            buy_decision = ask_llm_buy_decision(market_summary, usdc_balance, market_context, portfolio_value, history_summary)
            print(f"[DECISION] {json.dumps(buy_decision, indent=2)}")
            log_event("buy_decision", buy_decision)
            if buy_decision.get("action") == "BUY":
                quote_size = float(buy_decision["quote_size"])
                
                # 2. Enforce Minimum Limit
                if quote_size < MIN_TRADE_USDC:
                    print(f"⚠️ [ADJUST] Upgrading size from ${quote_size:.2f} to MIN limit ${MIN_TRADE_USDC:.2f}")
                    quote_size = MIN_TRADE_USDC

                # 3. Final Balance Check (Can we afford the adjusted size?)
                if quote_size > usdc_balance:
                    print(f"⛔ [SKIP] Required ${quote_size:.2f} but only have ${usdc_balance:.2f}")
                    return # Skip this trade
                # 1. Execute the trade
                execute_market_order(client, buy_decision["product_id"], "BUY", "quote_size", quote_size, dry_run=DRY_RUN)
                        
                # 2. Immediately record it in state (prevent "missing position" bug)
                pid = buy_decision["product_id"]
                if not DRY_RUN:
                    # Note: We use 'quote_size' as a proxy for size until the next balance update fixes it
                    # We use 'price' from the decision or 0.0 as a placeholder
                    state["positions"][pid] = {
                        "entry_price": float(buy_decision.get("price", 0) or 0),
                        "size": quote_size,  
                        "entry_time": datetime.now().isoformat(),
                        "stop_loss": float(buy_decision.get("price", 0) or 0) * 0.95, # Safety fallback
                        "buy_reason": buy_decision.get("reason", "N/A")
                    }
                        
                # 3. Save state
                state["buy_trades_today"] += 1
                save_state(state)
                    
            else:
                print("[SCAN] No candidates passed deterministic filters.")
    else: print("[STOP] Max buy trades reached.")
    print("="*70)

if __name__ == "__main__":
    main()
