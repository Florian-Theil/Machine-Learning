import gzip
import dpkt
import struct

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
                
                yield ts, ip.src, ip.dst, udp.sport, udp.dport, udp.data
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

if __name__ == "__main__":
    import sys
    from socket import inet_ntoa

    if len(sys.argv) < 2:
        print("Usage: python pcap2csv.py file.pcap[.gz]")
        sys.exit(1)
    

    filename = sys.argv[1]
    count = 0
    for ts, src, dst, sport, dport, payload in read_pcap_udp_payloads(filename):
        for msg_type, body in parse_iex_deep_payload(payload):
            msg = decode_trading_message(msg_type, body)
            if msg["type"] != "Unknown":
                count += 1
                print(msg)
        if count > 50:  # limit output for demo
            break

    print(f"\nDecoded {count} trading messages.")
