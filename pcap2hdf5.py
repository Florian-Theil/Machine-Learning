import gzip
import os
from pandas.core.arrays import base
import dpkt
import struct
from collections import defaultdict
import pandas as pd


def read_pcap_udp_payloads(filename):
    """
    Generator that yields (timestamp, src_ip, dst_ip, src_port, dst_port, payload_bytes)
    from UDP packets in a .pcap or .pcap.gz file.
    """
    # Decide whether to open normally or via gzip
    opener = gzip.open if filename.endswith(".gz") else open
    with opener(filename, "rb") as f:
        pcap = dpkt.pcapng.Reader(f)
        
        counter = 0
        for ts, buf in pcap:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                ip = eth.data
                if not isinstance(ip, dpkt.ip.IP):
                    continue
                udp = ip.data
                if not isinstance(udp, dpkt.udp.UDP):
                    continue
                if len(udp.data)==40:
                    continue
                
                yield ts, udp.data
            except Exception:
                # Skip malformed packets
                continue

def parse_iex_deep_payload(payload):
    """
    Yield (msg_type, msg_body) for each IEX DEEP message in the UDP payload.
    """
    # Skip 40-byte IEX-TP transport header
    offset = 40
    while offset + 3 <= len(payload):
        msg_len = struct.unpack_from("<H", payload, offset)[0]
        msg_type = chr(payload[offset + 2])
        body_start = offset + 3
        body_end = offset + 2 + msg_len
        if body_end > len(payload):
            break
        body = payload[body_start:body_end]
        
        yield msg_type, body
        offset = body_end

def decode_trading_message(msg_type, body):
    """
    Decode trading messages (types 8, 5, T, B, X, A) into Python dicts.
    """
    if msg_type in ('8', '5'):  # Price Level Update
        side = 'BUY' if msg_type == '8' else 'SELL'
        event_flags = body[0]
        ts, = struct.unpack_from("<Q", body, 1)
        symbol = body[9:17].decode('ascii').strip('\x00 ')
        size, = struct.unpack_from("<I", body, 17)
        price_int, = struct.unpack_from("<q", body, 21)
        price = price_int / 10000.0
        return dict(type='PriceLevelUpdate', side=side, ts=ts, symbol=symbol,
                    size=size, price=price, flags=event_flags)

    elif msg_type == 'T':  # Trade Report
        ts, = struct.unpack_from("<Q", body, 0)
        symbol = body[8:16].decode('ascii').strip('\x00 ')
        size, = struct.unpack_from("<I", body, 16)
        price_int, = struct.unpack_from("<q", body, 20)
        price = price_int / 10000.0
        trade_id, = struct.unpack_from("<Q", body, 28)
        return dict(type='TradeReport', ts=ts, symbol=symbol,
                    size=size, price=price, trade_id=trade_id)

    elif msg_type == 'B':  # Trade Break
        ts, = struct.unpack_from("<Q", body, 0)
        symbol = body[8:16].decode('ascii').strip('\x00 ')
        size, = struct.unpack_from("<I", body, 16)
        price_int, = struct.unpack_from("<q", body, 20)
        price = price_int / 10000.0
        trade_id, = struct.unpack_from("<Q", body, 28)
        return dict(type='TradeBreak', ts=ts, symbol=symbol,
                    size=size, price=price, trade_id=trade_id)

    elif msg_type == 'X':  # Official Price
        ts, = struct.unpack_from("<Q", body, 0)
        symbol = body[8:16].decode('ascii').strip('\x00 ')
        price_int, = struct.unpack_from("<q", body, 16)
        price = price_int / 10000.0
        return dict(type='OfficialPrice', ts=ts, symbol=symbol, price=price)

    elif msg_type == 'A':  # Auction Information
        ts, = struct.unpack_from("<Q", body, 0)
        symbol = body[8:16].decode('ascii').strip('\x00 ')
        ref_price_int, = struct.unpack_from("<q", body, 16)
        auction_clearing_price_int, = struct.unpack_from("<q", body, 24)
        imbalance_size, = struct.unpack_from("<I", body, 32)
        ref_price = ref_price_int / 10000.0
        auction_price = auction_clearing_price_int / 10000.0
        return dict(type='AuctionInfo', ts=ts, symbol=symbol,
                    ref_price=ref_price, auction_price=auction_price,
                    imbalance=imbalance_size)

    else:
        return dict(type='Unknown', raw=body.hex())

# ---------------------------------------------------------------------
# 3️⃣ ORDER BOOK STATE MANAGEMENT
# ---------------------------------------------------------------------

class OrderBookManager:
    """
    Maintains a minimal in-memory order book (top-of-book only) per symbol.
    Updates occur with every PriceLevelUpdate message.
    """

    def __init__(self):
        self.bids = defaultdict(dict)  # symbol → {price: size}
        self.asks = defaultdict(dict)

    def update(self, msg):
        """
        Update the internal bid/ask dictionaries and, if both sides exist,
        return a record of derived features (mid, spread, imbalance, etc.)
        suitable for storage or training.
        """
        if not msg or msg["type"] != "PriceLevelUpdate":
            return None

        sym, side, price, size, ts = (
            msg["symbol"], msg["side"], msg["price"], msg["size"], msg["ts"]
        )

        # Choose which side of the book to update
        book = self.bids if side == "BUY" else self.asks

        # Update or remove this price level
        if size == 0:
            book[sym].pop(price, None)
        else:
            book[sym][price] = size

        # If both sides exist, compute features
        if self.bids[sym] and self.asks[sym]:
            best_bid = max(self.bids[sym])
            best_ask = min(self.asks[sym])
            bid_size = self.bids[sym][best_bid]
            ask_size = self.asks[sym][best_ask]

            # Derived quantities
            mid = 0.5 * (best_bid + best_ask)
            spread = best_ask - best_bid
            imbalance = (bid_size - ask_size) / (bid_size + ask_size)

            return dict(
                ts=ts,
                symbol=sym,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_size=bid_size,
                ask_size=ask_size,
                mid=mid,
                spread=spread,
                imbalance=imbalance,
            )
        return None


# ---------------------------------------------------------------------
# 4️⃣ RESAMPLING TO FIXED FREQUENCY
# ---------------------------------------------------------------------

def resample(df, freq="100ms"):
    """
    Convert an irregular stream of order-book updates into a
    fixed-frequency (e.g. every 100 ms) time series.

    For each symbol:
      - take the last known state within each 100 ms window,
      - forward-fill missing intervals to keep book continuity.
    """
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ns", errors="coerce")
    df = (
        df.set_index("timestamp")
          .groupby("symbol")
          .resample(freq)
          .last()
          .ffill()              # propagate latest known state
          .drop(columns="symbol")
          .reset_index()
    )
    return df


# ---------------------------------------------------------------------
# 5️⃣ MAIN STREAMING PIPELINE
# ---------------------------------------------------------------------

def stream_to_hdf5(pcap_file, h5_path, chunk_size=500000, freq="100ms"):
    """
    Read packets from `pcap_file`, decode IEX DEEP updates, compute
    top-of-book features, resample to fixed frequency, and append to
    an HDF5 file incrementally.

    Parameters
    ----------
    pcap_file : str
        Path to .pcapng or .pcap input file.
    h5_path : str
        Path to output .h5 file.
    chunk_size : int, optional
        Number of decoded rows to accumulate before writing (default 10000).
    freq : str, optional
        Resampling frequency, pandas-style (default '100ms').
    """
    stop_after = pd.Timedelta("365days")  # for testing, limit to 1 year
    # stop_after = pd.Timedelta("365days")  # for full file processing
    start_ts = None
    
    mgr = OrderBookManager()        # maintains per-symbol order books
    recs = []                       # in-memory buffer for decoded records
    total_rows = 0

    # Open the HDF5 file (compressed with Blosc)
    base, ext = os.path.splitext(h5_path)
    temp_file = f"{base}_temp{ext}"
    store = pd.HDFStore(temp_file, mode="w", complevel=2, complib="blosc")

    # Main streaming loop over packets
    for ts, payload in read_pcap_udp_payloads(pcap_file):
        for msg_type, body in parse_iex_deep_payload(payload):
            msg = decode_trading_message(msg_type, body)
            rec = mgr.update(msg)
            if rec:
                recs.append(rec)
                
                if start_ts is None:
                    start_ts = pd.to_datetime(rec["ts"], unit="ns")
                current_ts = pd.to_datetime(rec["ts"], unit="ns")
        # Check time limit
        if start_ts and current_ts - start_ts > stop_after:
            print(f"Reached {stop_after} time of data, stopping.")
            break

        # Once we reach chunk_size, convert to DataFrame and write
        if len(recs) >= chunk_size:
            df = pd.DataFrame.from_records(recs)
            # df_rs = resample_chunk(df, freq=freq)
            store.append("deep", df, format="table", data_columns=["symbol"])
            total_rows += len(df)
            recs.clear()
            print(f"Written {total_rows:,} rows so far...")

    # Write any leftover records
    if recs:
        df = pd.DataFrame.from_records(recs)
        # df_rs = resample_chunk(df, freq=freq)
        store.append("deep", df, format="table", data_columns=["symbol"])
        total_rows += len(df)

    store.close()
    # Now resample the data in the HDF5 file
    print("Resampling data to fixed frequency...")
    df = pd.read_hdf(temp_file, "deep")
    df_rs = resample(df, freq="100ms")
    df_rs.to_hdf(h5_path, key="deep", format="table")
    
    print(f"✅ Finished: {total_rows:,} rows saved to {h5_path}")
        
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pcap2hdf5.py input.pcap[.gz] output.h5")
        sys.exit(1)
    
    pcapfile = sys.argv[1]
    hdf5file = sys.argv[2]
    count = 0
    
    stream_to_hdf5(pcapfile, hdf5file, chunk_size=500000)