"""
blockchain.py — Lightweight simulated blockchain for FIR Chargesheet integrity.

No external node required. Uses SHA-256 chaining so every FIR / chargesheet
record is cryptographically linked. Any tampering breaks the chain.
"""

import hashlib
import json
import time
import os
from datetime import datetime
from typing import List, Dict, Optional


# ──────────────────────────────────────────────
# Block
# ──────────────────────────────────────────────

class Block:
    def __init__(self, index: int, data: Dict, previous_hash: str, timestamp: float = None):
        self.index         = index
        self.timestamp     = timestamp or time.time()
        self.data          = data
        self.previous_hash = previous_hash
        self.nonce         = 0
        self.hash          = self._compute_hash()

    # ── internal ──────────────────────────────
    def _compute_hash(self) -> str:
        block_string = json.dumps(
            {
                "index":         self.index,
                "timestamp":     self.timestamp,
                "data":          self.data,
                "previous_hash": self.previous_hash,
                "nonce":         self.nonce,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(block_string.encode()).hexdigest()

    def proof_of_work(self, difficulty: int = 2) -> None:
        """Simple proof-of-work (difficulty = leading zeros)."""
        target = "0" * difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self._compute_hash()

    # ── serialisation ─────────────────────────
    def to_dict(self) -> Dict:
        return {
            "index":         self.index,
            "timestamp":     self.timestamp,
            "data":          self.data,
            "previous_hash": self.previous_hash,
            "nonce":         self.nonce,
            "hash":          self.hash,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Block":
        b = cls(
            index         = d["index"],
            data          = d["data"],
            previous_hash = d["previous_hash"],
            timestamp     = d["timestamp"],
        )
        b.nonce = d["nonce"]
        b.hash  = d["hash"]
        return b


# ──────────────────────────────────────────────
# Blockchain
# ──────────────────────────────────────────────

class FIRBlockchain:
    CHAIN_FILE  = os.path.join("models", "fir_blockchain.json")
    DIFFICULTY  = 2          # keep low so the web request stays fast

    def __init__(self):
        os.makedirs("models", exist_ok=True)
        self.chain: List[Block] = []
        self._load_or_create()

    # ── genesis / persistence ─────────────────
    def _create_genesis(self) -> Block:
        genesis = Block(
            index         = 0,
            data          = {"type": "GENESIS", "message": "FIR Blockchain Initialized"},
            previous_hash = "0" * 64,
        )
        genesis.proof_of_work(self.DIFFICULTY)
        return genesis

    def _load_or_create(self):
        if os.path.exists(self.CHAIN_FILE):
            try:
                with open(self.CHAIN_FILE, "r") as f:
                    raw = json.load(f)
                self.chain = [Block.from_dict(b) for b in raw]
                print(f"[Blockchain] Loaded {len(self.chain)} blocks.")
                return
            except Exception as e:
                print(f"[Blockchain] Load failed ({e}), recreating.")
        self.chain = [self._create_genesis()]
        self._persist()

    def _persist(self):
        with open(self.CHAIN_FILE, "w") as f:
            json.dump([b.to_dict() for b in self.chain], f, indent=2, default=str)

    # ── public API ────────────────────────────
    @property
    def last_block(self) -> Block:
        return self.chain[-1]

    def add_block(self, data: Dict) -> Block:
        block = Block(
            index         = len(self.chain),
            data          = data,
            previous_hash = self.last_block.hash,
        )
        block.proof_of_work(self.DIFFICULTY)
        self.chain.append(block)
        self._persist()
        return block

    # ── FIR-specific helpers ──────────────────
    @staticmethod
    def hash_document(data: Dict) -> str:
        """Deterministic SHA-256 of any dict (used to fingerprint FIR / chargesheet)."""
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

    def register_fir(self, fir_id: int, fir_number: str, complainant: str,
                     description: str, officer: str) -> Block:
        payload = {
            "type":        "FIR_CREATED",
            "fir_id":      fir_id,
            "fir_number":  fir_number,
            "complainant": complainant,
            "officer":     officer,
            "doc_hash":    self.hash_document(
                               {"fir_number": fir_number,
                                "complainant": complainant,
                                "description": description}
                           ),
            "timestamp":   datetime.utcnow().isoformat(),
        }
        return self.add_block(payload)

    def register_evidence(self, fir_id: int, fir_number: str,
                          evidence_desc: str, uploaded_by: str) -> Block:
        payload = {
            "type":         "EVIDENCE_ADDED",
            "fir_id":       fir_id,
            "fir_number":   fir_number,
            "evidence_hash": self.hash_document({"desc": evidence_desc}),
            "uploaded_by":  uploaded_by,
            "timestamp":    datetime.utcnow().isoformat(),
        }
        return self.add_block(payload)

    def register_chargesheet(self, fir_id: int, fir_number: str,
                             ipc_sections: str, generated_by: str,
                             chargesheet_data: Dict) -> Block:
        payload = {
            "type":             "CHARGESHEET_GENERATED",
            "fir_id":           fir_id,
            "fir_number":       fir_number,
            "ipc_sections":     ipc_sections,
            "generated_by":     generated_by,
            "chargesheet_hash": self.hash_document(chargesheet_data),
            "timestamp":        datetime.utcnow().isoformat(),
        }
        return self.add_block(payload)

    def register_status_change(self, fir_id: int, fir_number: str,
                                old_status: str, new_status: str,
                                changed_by: str) -> Block:
        payload = {
            "type":       "STATUS_CHANGED",
            "fir_id":     fir_id,
            "fir_number": fir_number,
            "old_status": old_status,
            "new_status": new_status,
            "changed_by": changed_by,
            "timestamp":  datetime.utcnow().isoformat(),
        }
        return self.add_block(payload)

    # ── integrity ─────────────────────────────
    def is_chain_valid(self) -> bool:
        for i in range(1, len(self.chain)):
            curr = self.chain[i]
            prev = self.chain[i - 1]
            if curr.hash != curr._compute_hash():
                return False
            if curr.previous_hash != prev.hash:
                return False
        return True

    def get_fir_audit_trail(self, fir_id: int) -> List[Dict]:
        return [
            b.to_dict()
            for b in self.chain
            if b.data.get("fir_id") == fir_id
        ]

    def get_all_blocks(self) -> List[Dict]:
        return [b.to_dict() for b in self.chain]

    def chain_stats(self) -> Dict:
        return {
            "total_blocks":   len(self.chain),
            "chain_valid":    self.is_chain_valid(),
            "last_block_hash": self.last_block.hash,
            "genesis_hash":    self.chain[0].hash,
        }


# ── module-level singleton ────────────────────
_blockchain: Optional[FIRBlockchain] = None


def get_blockchain() -> FIRBlockchain:
    global _blockchain
    if _blockchain is None:
        _blockchain = FIRBlockchain()
    return _blockchain
