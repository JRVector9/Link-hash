import time
import requests
from typing import Any, Dict, List, Optional, Tuple

DEX_BASE = "https://api.dexscreener.com"


# -----------------------------
# HTTP / JSON
# -----------------------------
def get_json(url: str, timeout: int = 15) -> Any:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _to_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


# -----------------------------
# Discovery
# -----------------------------
def discover_tokens_solana(limit: int = 200) -> List[str]:
    """
    DexScreener 최신/부스트/탑 소스에서 Solana 토큰 주소 후보를 수집.
    """
    sources = [
        f"{DEX_BASE}/token-profiles/latest/v1",
        f"{DEX_BASE}/token-boosts/latest/v1",
        f"{DEX_BASE}/token-boosts/top/v1",
    ]

    tokens: List[str] = []
    for url in sources:
        try:
            data = get_json(url)
        except Exception:
            continue

        items = (
            data
            if isinstance(data, list)
            else data.get("tokens")
            or data.get("pairs")
            or data.get("data")
            or data
        )
        if not isinstance(items, list):
            continue

        for it in items:
            if not isinstance(it, dict):
                continue
            chain = it.get("chainId") or it.get("chain") or it.get("chain_id")
            addr = it.get("tokenAddress") or it.get("address") or it.get("token_address")
            if chain == "solana" and addr:
                tokens.append(addr)

        time.sleep(0.2)

    uniq: List[str] = []
    seen = set()
    for a in tokens:
        if a in seen:
            continue
        seen.add(a)
        uniq.append(a)
        if len(uniq) >= limit:
            break
    return uniq


# -----------------------------
# Pair selection (important!)
# -----------------------------
def _pair_liquidity_usd(p: Dict[str, Any]) -> float:
    return float((p.get("liquidity") or {}).get("usd") or 0)


def _pair_volume_24h(p: Dict[str, Any]) -> float:
    return float((p.get("volume") or {}).get("h24") or 0)


def _pair_txns_24h(p: Dict[str, Any]) -> int:
    tx = (p.get("txns") or {}).get("h24") or {}
    if not isinstance(tx, dict):
        return 0
    buys = tx.get("buys") or 0
    sells = tx.get("sells") or 0
    try:
        return int(buys) + int(sells)
    except Exception:
        return 0


def _sanity_market_cap(mcap: Optional[float], fdv: Optional[float]) -> Optional[float]:
    """
    시장에서 자주 터지는 케이스 방어:
    - marketCap이 fdv보다 의미있게 큰 경우(공급/가격 추정 꼬임) marketCap을 신뢰하지 않음(None 처리)
    - mcap이 음수/0도 None 처리
    """
    if mcap is None:
        return None
    if mcap <= 0:
        return None
    if fdv is not None and fdv > 0:
        # marketCap이 fdv보다 20% 이상 큰 케이스는 비정상으로 간주
        if mcap > fdv * 1.2:
            return None
    return mcap


def _pick_best_pair(pairs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    기존: liquidity만으로 1등 pair 선택
    개선: (LP + Volume) 가중치 기반으로 선택하여 "이상한 pair" 확률을 낮춤.
    """
    if not pairs:
        return None

    # 1) 후보 정리: dict만
    cleaned = [p for p in pairs if isinstance(p, dict)]
    if not cleaned:
        return None

    # 2) 점수 계산: LP 0.7 + Vol24 0.3
    def score(p: Dict[str, Any]) -> float:
        lp = _pair_liquidity_usd(p)
        vol = _pair_volume_24h(p)
        return lp * 0.7 + vol * 0.3

    cleaned.sort(key=score, reverse=True)
    return cleaned[0]


def best_pair_snapshot(
    chain_id: str,
    token_address: str,
    *,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    토큰의 pair들 중 가장 "안정적인" 대표 pair를 고르고 스냅샷 생성.
    - LP+Vol로 pair 선택
    - marketCap sanity 체크 (fdv 대비 이상치 제거)
    """
    url = f"{DEX_BASE}/token-pairs/v1/{chain_id}/{token_address}"

    try:
        pairs = get_json(url)
    except Exception:
        return None

    if not pairs or not isinstance(pairs, list):
        return None

    p = _pick_best_pair(pairs)
    if not p:
        return None

    txns24h = _pair_txns_24h(p)

    fdv = _to_float(p.get("fdv"))
    mcap_raw = _to_float(p.get("marketCap"))
    mcap = _sanity_market_cap(mcap_raw, fdv)

    snap = {
        "token_address": token_address,
        "pair_address": p.get("pairAddress"),
        "dex_id": p.get("dexId"),
        "base_symbol": (p.get("baseToken") or {}).get("symbol"),
        "price_usd": _to_float(p.get("priceUsd")),
        "liquidity_usd": _to_float((p.get("liquidity") or {}).get("usd")),
        "volume_24h": _to_float((p.get("volume") or {}).get("h24")),
        "txns_24h": txns24h,
        "fdv": fdv,
        "market_cap": mcap,  # ✅ sanity 적용된 값
        "url": p.get("url"),
    }

    if debug:
        lp = (p.get("liquidity") or {}).get("usd")
        vol = (p.get("volume") or {}).get("h24")
        print(
            "[DexScreener Pair]",
            snap.get("base_symbol"),
            "token=", token_address,
            "pair=", p.get("pairAddress"),
            "dex=", p.get("dexId"),
            "lp=", lp,
            "vol24=", vol,
            "tx24=", txns24h,
            "mcap_raw=", mcap_raw,
            "fdv=", fdv,
            "mcap_used=", mcap,
            "url=", p.get("url"),
        )

    return snap


# -----------------------------
# Filters / scoring
# -----------------------------
def compute_filters(snap: Dict[str, Any]) -> Dict[str, Any]:
    """
    Potential score 산정용 파생지표 계산.
    """
    vol = snap.get("volume_24h")
    mcap = snap.get("market_cap")
    lp = snap.get("liquidity_usd")

    vol_mcap_ratio = None
    lp_mcap_ratio = None

    if vol is not None and mcap is not None and mcap > 0:
        vol_mcap_ratio = vol / mcap
    if lp is not None and mcap is not None and mcap > 0:
        lp_mcap_ratio = lp / mcap

    txns = snap.get("txns_24h")

    cond1 = (txns is not None and txns >= 200)
    cond2 = (vol_mcap_ratio is not None and vol_mcap_ratio >= 0.30)
    cond3 = (vol is not None and vol >= 10_000)
    cond4 = (lp_mcap_ratio is not None and lp_mcap_ratio >= 0.05)

    potential_score_hits = int(cond1) + int(cond2) + int(cond3) + int(cond4)

    snap["vol_mcap_ratio"] = vol_mcap_ratio
    snap["lp_mcap_ratio"] = lp_mcap_ratio
    snap["potential_score_hits"] = potential_score_hits
    return snap

def get_lp_mcap_pct(chain_id: str, token_address: str):
    """
    DexScreener에서 best pair 기준으로
    - liquidity_usd, market_cap, lp_mcap_pct 계산해서 반환
    """
    snap = best_pair_snapshot(chain_id, token_address)
    if not snap:
        return None

    lp = snap.get("liquidity_usd")
    mcap = snap.get("market_cap")

    lp_mcap_pct = None
    if lp is not None and mcap is not None and mcap > 0:
        lp_mcap_pct = round((lp / mcap) * 100, 2)

    return {
        "liquidity_usd": lp,
        "market_cap": mcap,
        "lp_mcap_pct": lp_mcap_pct,
    }
