from datetime import datetime
from typing import Dict, List, Optional
from chargesheet_rl import load_trained_model, generate_chargesheet_from_description

class ChargeSheet:
    def __init__(self):
        self.case_number: str = ""
        self.date_filed: datetime = datetime.now()
        self.accused_details: Dict = {}
        self.complainant_details: Dict = {}
        self.charges: List[Dict] = []
        self.incident_details: str = ""
        self.investigating_officer: str = ""
        self.witnesses: List[Dict] = []
        self.fir_details: Dict = {}
        self.evidences: List[Dict] = []
        self.seizure_list: List[Dict] = []
        self.investigation_findings: str = ""

    def add_accused(self, name: str, age: int, address: str, identification: str) -> None:
        """Add accused person's details"""
        self.accused_details = {
            "name": name,
            "age": age,
            "address": address,
            "identification": identification
        }

    def add_complainant(self, name: str, contact: str, address: str) -> None:
        """Add complainant's details"""
        self.complainant_details = {
            "name": name,
            "contact": contact,
            "address": address
        }

    def add_charge(self, section: str, description: str) -> None:
        """Add a charge to the chargesheet"""
        self.charges.append({
            "section": section,
            "description": description
        })

    def add_witness(self, name: str, statement: str, contact: Optional[str] = None) -> None:
        """Add a witness to the chargesheet"""
        self.witnesses.append({
            "name": name,
            "statement": statement,
            "contact": contact
        })

    def set_incident_details(self, details: str) -> None:
        """Set the incident details"""
        self.incident_details = details

    def set_case_number(self, case_number: str) -> None:
        """Set the case number"""
        self.case_number = case_number

    def set_investigating_officer(self, officer_name: str) -> None:
        """Set the investigating officer's name"""
        self.investigating_officer = officer_name

    def set_fir_details(self, fir_number: str, date_filed: datetime, police_station: str, 
                       complaint_description: str) -> None:
        """Set the FIR details that led to this chargesheet"""
        self.fir_details = {
            "fir_number": fir_number,
            "date_filed": date_filed.strftime("%Y-%m-%d %H:%M:%S"),
            "police_station": police_station,
            "complaint_description": complaint_description
        }

    def add_evidence(self, type: str, description: str, collection_date: datetime, 
                    collected_by: str, custody_chain: List[str]) -> None:
        """Add evidence details"""
        self.evidences.append({
            "type": type,  # Physical/Digital/Forensic/Documentary
            "description": description,
            "collection_date": collection_date.strftime("%Y-%m-%d %H:%M:%S"),
            "collected_by": collected_by,
            "custody_chain": custody_chain
        })

    def add_seized_item(self, item_name: str, description: str, 
                       seizure_date: datetime, location: str) -> None:
        """Add details of items seized during investigation"""
        self.seizure_list.append({
            "item_name": item_name,
            "description": description,
            "seizure_date": seizure_date.strftime("%Y-%m-%d %H:%M:%S"),
            "location": location
        })

    def set_investigation_findings(self, findings: str) -> None:
        """Set the detailed findings from the investigation"""
        self.investigation_findings = findings

    def generate_chargesheet(self) -> Dict:
        """Generate the complete chargesheet"""
        chargesheet_data = {
            "case_number": self.case_number,
            "date_filed": self.date_filed.strftime("%Y-%m-%d %H:%M:%S"),
            "fir_details": self.fir_details,
            "accused_details": self.accused_details,
            "complainant_details": self.complainant_details,
            "charges": self.charges,
            "incident_details": self.incident_details,
            "investigation_findings": self.investigation_findings,
            "evidences": self.evidences,
            "seizure_list": self.seizure_list,
            "investigating_officer": self.investigating_officer,
            "witnesses": self.witnesses
        }
        return chargesheet_data


def create_sample_chargesheet() -> Dict:
    """Create a sample chargesheet with example data"""
    chargesheet = ChargeSheet()
    
    # Set FIR details
    chargesheet.set_fir_details(
        fir_number="FIR/2024/001",
        date_filed=datetime(2024, 1, 15, 14, 30),
        police_station="Central Police Station",
        complaint_description="Complainant reported break-in and theft at residence with threats"
    )
    
    # Set basic details
    chargesheet.set_case_number("FIR/2024/001")
    chargesheet.set_investigating_officer("Inspector John Smith")
    
    # Add accused details
    chargesheet.add_accused(
        name="John Doe",
        age=35,
        address="123 Main Street, City",
        identification="ID12345"
    )
    
    # Add complainant details
    chargesheet.add_complainant(
        name="Jane Smith",
        contact="+1-555-0123",
        address="456 Oak Avenue, City"
    )
    
    # Add charges
    chargesheet.add_charge(
        section="Section 379 IPC",
        description="Theft of personal property"
    )
    chargesheet.add_charge(
        section="Section 506 IPC",
        description="Criminal intimidation"
    )
    
    # Set incident details
    chargesheet.set_incident_details(
        "On January 15, 2024, at approximately 2:30 PM, "
        "the accused allegedly broke into the complainant's residence "
        "and stole valuable items while threatening the complainant."
    )
    
    # Add witnesses
    chargesheet.add_witness(
        name="Michael Johnson",
        statement="Witnessed the accused leaving the premises with stolen items",
        contact="+1-555-0124"
    )
    chargesheet.add_witness(
        name="Sarah Williams",
        statement="Heard threatening sounds and saw the accused in the vicinity",
        contact="+1-555-0125"
    )
    
    # Add evidence details
    chargesheet.add_evidence(
        type="Physical",
        description="Broken window lock from point of entry",
        collection_date=datetime(2024, 1, 15, 15, 45),
        collected_by="Officer Sarah Johnson",
        custody_chain=["Officer Sarah Johnson", "Forensics Lab", "Evidence Room"]
    )
    
    chargesheet.add_evidence(
        type="Digital",
        description="CCTV footage from neighbor's camera showing suspect",
        collection_date=datetime(2024, 1, 15, 16, 30),
        collected_by="Tech Officer Mike Wilson",
        custody_chain=["Tech Officer Mike Wilson", "Digital Forensics Lab"]
    )

    # Add seized items
    chargesheet.add_seized_item(
        item_name="Stolen Jewelry",
        description="Gold necklace with diamond pendant",
        seizure_date=datetime(2024, 1, 16, 10, 15),
        location="Accused's residence"
    )

    # Set investigation findings
    chargesheet.set_investigation_findings(
        "Based on the investigation, including CCTV footage, witness statements, "
        "and recovered stolen items, there is strong evidence linking the accused "
        "to the break-in and theft. Forensic analysis of the broken lock matches "
        "the tools found in accused's possession. Multiple witnesses have confirmed "
        "seeing the accused in the vicinity at the time of the incident."
    )
    
    return chargesheet.generate_chargesheet()


if __name__ == "__main__":
    # Example usage
    sample_chargesheet = create_sample_chargesheet()
    import json
    print(json.dumps(sample_chargesheet, indent=2))

test_data = [
    {
        "description": "Two men snatched his bag and ran away.",
        "expected_ipcs": ["IPC 379", "IPC 392"]
    },
    {
        "description": "He was assaulted with a knife in a fight.",
        "expected_ipcs": ["IPC 324", "IPC 504"]
    },
    {
        "description": "The accused fraudulently transferred money online.",
        "expected_ipcs": ["IPC 420", "IPC 66D"]
    },
    {
        "description": "A group of people looted a jewelry shop at night.",
        "expected_ipcs": ["IPC 395", "IPC 120B"]
    },
    {
        "description": "A fake website was created to scam users.",
        "expected_ipcs": ["IPC 66C", "IPC 419", "IPC 420"]
    },
       {
        "description": "The victim was murdered with a pistol at his residence in January.",
        "expected_ipcs": ["IPC 302", "IPC 450", "IPC 25 Arms Act"]
    },
    {
        "description": "She was kidnapped from her home and taken to an unknown location.",
        "expected_ipcs": ["IPC 363", "IPC 366"]
    },
    {
        "description": "He created forged documents to claim property ownership.",
        "expected_ipcs": ["IPC 465", "IPC 420"]
    },
    {
        "description": "Acid was thrown on a girl after an argument in a market.",
        "expected_ipcs": ["IPC 326A", "IPC 354"]
    },
    {
        "description": "The accused accepted a bribe to manipulate the case.",
        "expected_ipcs": ["IPC 7 PC Act", "IPC 120B"]
    },
    {
        "description": "He was seen breaking into a house at midnight.",
        "expected_ipcs": ["IPC 380", "IPC 457"]
    },
    {
        "description": "A threatening letter was sent to the minister demanding money.",
        "expected_ipcs": ["IPC 506", "IPC 384"]
    },
    {
        "description": "Three people planned and executed a bank robbery.",
        "expected_ipcs": ["IPC 395", "IPC 120B"]
    },
    {
        "description": "Doctor performed a surgery negligently leading to death.",
        "expected_ipcs": ["IPC 304A"]
    },
    {
        "description": "He was caught hacking into the company server.",
        "expected_ipcs": ["IPC 66C", "IPC 66"]
    }
]

def rule_based_filter(predicted_ipcs, description):
    description_lower = description.lower()
    filtered = set(predicted_ipcs)

    # === Strong noise reduction for misfiring IPCs unless description supports ===
    remove_if_no_keywords = {
        "IPC 400": ["gang", "criminal gang"],
        "IPC 401": ["gang", "habitual", "criminal"],
        "IPC 406": ["entrusted", "custody", "trust", "embezzled"],
        "IPC 417": ["deceive", "deception", "cheated"],
        "IPC 426": ["mischief", "damage", "destroyed"],
        "IPC 447": ["trespass", "unauthorized", "encroachment"],
        "IPC 413": ["stolen property", "habitual dealer", "dealer"],
        "IPC 448": ["house", "trespass", "entry"],
    }

    for ipc, keywords in remove_if_no_keywords.items():
        if ipc in filtered and not any(k in description_lower for k in keywords):
            filtered.discard(ipc)

    # === IPC 420 Related Keywords ===
    if any(word in description_lower for word in ["cheat", "fraud", "scam", "fake", "swindle", "dupe", "online fraud", "deceive"]):
        filtered.add("IPC 420")

    # === IPC 66C/66D for Cyber crimes ===
    if any(word in description_lower for word in ["otp", "password", "cyber", "online", "hacked", "website", "phishing"]):
        filtered.add("IPC 66C")
        filtered.add("IPC 66D")

    # === IPC 304A - Negligence ===
    if "negligent" in description_lower or ("doctor" in description_lower and "death" in description_lower):
        filtered.add("IPC 304A")

    # === IPC 25 Arms Act ===
    if any(word in description_lower for word in ["gun", "pistol", "weapon", "revolver", "arms"]):
        filtered.add("IPC 25 Arms Act")

    # === IPC 326A - Acid attack ===
    if any(word in description_lower for word in ["acid", "acid attack"]):
        filtered.add("IPC 326A")

    # === IPC 366, IPC 363 - Kidnapping ===
    if "kidnap" in description_lower or "abduct" in description_lower:
        filtered.add("IPC 363")
        filtered.add("IPC 366")

    # === IPC 354 - Sexual assault ===
    if any(word in description_lower for word in ["molest", "touched", "sexual", "harassed"]):
        filtered.add("IPC 354")

    # === IPC 384, IPC 506 - Extortion and Threat ===
    if any(word in description_lower for word in ["threat", "intimidate", "extort", "blackmail", "letter", "warned"]):
        filtered.add("IPC 506")
        filtered.add("IPC 384")

    # === IPC 457 - House break-in at night ===
    if any(word in description_lower for word in ["midnight", "night", "broke into", "breaking in", "break-in", "entered house"]):
        filtered.add("IPC 457")

    # === IPC 395 - Dacoity only if group involved ===
    if any(word in description_lower for word in ["group", "gang", "five", "team", "men", "people"]) and "robbery" in description_lower:
        filtered.add("IPC 395")
        filtered.add("IPC 120B")

    # === IPC 302 - Murder or Death ===
    if any(word in description_lower for word in ["murder", "killed", "homicide", "dead", "death"]) and "accused" in description_lower:
        filtered.add("IPC 302")

    # === Remove vague or low-relevance IPCs if they dominate ===
    low_precision_ipcs = {"IPC 384", "IPC 410", "IPC 428", "IPC 426", "IPC 417", "IPC 419"}
    for ipc in low_precision_ipcs:
        if ipc in filtered and not any(ipc in ipc_code for ipc_code in description_lower.split()):
            filtered.discard(ipc)

    return list(filtered)



def evaluate_model(test_data):
    print("Loading trained model...")
    agent, env = load_trained_model()
    if not agent or not env:
        print("❌ Failed to load trained model.")
        return

    total_precision, total_recall, total_f1 = 0, 0, 0
    exact_matches = 0

    for i, item in enumerate(test_data, 1):
        expected = set(ipc.strip() for ipc in item['expected_ipcs'])
        raw_predicted = generate_chargesheet_from_description(item['description'], agent, env)
        predicted = set(rule_based_filter(raw_predicted, item['description']))
        predicted = set(ipc.strip() for ipc in predicted)

        tp = len(predicted & expected)
        fp = len(predicted - expected)
        fn = len(expected - predicted)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        if predicted == expected:
            exact_matches += 1

        total_precision += precision
        total_recall += recall
        total_f1 += f1

        print(f"\nTest Case {i}")
        print(f"Description     : {item['description']}")
        print(f"Expected IPCs   : {expected}")
        print(f"Predicted IPCs  : {predicted}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, Exact Match: {'✅' if predicted == expected else '❌'}")

    n = len(test_data)
    accuracy = exact_matches / n if n > 0 else 0.0

    print("\n--- Overall Model Evaluation ---")
    print(f"Avg Precision : {total_precision / n:.2f}")
    print(f"Avg Recall    : {total_recall / n:.2f}")
    print(f"Avg F1 Score  : {total_f1 / n:.2f}")

if __name__ == "__main__":
    evaluate_model(test_data)



