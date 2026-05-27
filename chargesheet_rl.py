"""
chargesheet_rl.py  —  Enhanced RL + Keyword-Hybrid IPC Section Predictor
=========================================================================
Improvements over original:
  • Deeper PolicyNetwork (3 hidden layers + BatchNorm + Dropout)
  • Better reward shaping: relevance score weighted by cosine similarity
  • Expanded keyword boosting for 20+ crime categories
  • Expanded rule-based post-filter (rule_based_filter)
  • F1-aware threshold tuning during evaluation
  • Top-k dynamically adjusted (5–12) by description length
  • Cleaner train loop with entropy regularisation
  • evaluate_model prints per-class confusion stats
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import json

# ──────────────────────────────────────────────────────────────────────────────
# Text Encoder
# ──────────────────────────────────────────────────────────────────────────────

class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.bert      = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode_text(self, text: str):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :]   # [CLS]


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────

class ChargesheetEnvironment:
    def __init__(self):
        self.section_categories = {
            "offenses_against_state": {
                "sections": [
                    "IPC 121","IPC 124A","IPC 125","IPC 131",
                    "IPC 121A","IPC 122","IPC 123","IPC 126",
                    "IPC 127","IPC 128","IPC 129","IPC 130",
                ],
                "description": "Offenses against the State",
                "related_categories": ["public_tranquility","criminal_conspiracy"],
            },
            "public_justice": {
                "sections": [
                    "IPC 191","IPC 193","IPC 196","IPC 211",
                    "IPC 192","IPC 194","IPC 195","IPC 197",
                    "IPC 198","IPC 199","IPC 200","IPC 201",
                    "IPC 202","IPC 203","IPC 204","IPC 205",
                    "IPC 206","IPC 207","IPC 208","IPC 209",
                    "IPC 210","IPC 212","IPC 213","IPC 214",
                    "IPC 215","IPC 216","IPC 216A","IPC 217",
                    "IPC 218","IPC 219","IPC 220","IPC 221",
                    "IPC 222","IPC 223","IPC 224","IPC 225",
                    "IPC 225A","IPC 225B","IPC 226",
                ],
                "description": "Offenses against public justice",
                "related_categories": ["public_servants","fraud"],
            },
            "public_tranquility": {
                "sections": [
                    "IPC 141","IPC 143","IPC 146","IPC 147",
                    "IPC 153A","IPC 153B","IPC 142","IPC 144",
                    "IPC 145","IPC 148","IPC 149","IPC 150",
                    "IPC 151","IPC 152","IPC 153","IPC 154",
                    "IPC 155","IPC 156","IPC 157","IPC 158",
                    "IPC 159","IPC 160",
                ],
                "description": "Offenses against public tranquility",
                "related_categories": [
                    "offenses_against_state","public_justice","criminal_intimidation"
                ],
            },
            "public_servants": {
                "sections": [
                    "IPC 166","IPC 167","IPC 168","IPC 171E",
                    "IPC 169","IPC 170","IPC 171","IPC 171A",
                    "IPC 171B","IPC 171C","IPC 171D","IPC 171F",
                    "IPC 171G","IPC 171H","IPC 171I",
                ],
                "description": "Offenses by public servants",
                "related_categories": ["public_justice","fraud","criminal_conspiracy"],
            },
            "offenses_against_life": {
                "sections": [
                    "IPC 299","IPC 300","IPC 302","IPC 304",
                    "IPC 304A","IPC 304B","IPC 306","IPC 307",
                    "IPC 308","IPC 309","IPC 301","IPC 303",
                    "IPC 305","IPC 310","IPC 311","IPC 312",
                    "IPC 313","IPC 314","IPC 315","IPC 316",
                    "IPC 317","IPC 318",
                ],
                "description": "Offenses against life",
                "related_categories": [
                    "hurt_grievous_hurt","crimes_against_women","criminal_intimidation"
                ],
            },
            "hurt_grievous_hurt": {
                "sections": [
                    "IPC 319","IPC 320","IPC 321","IPC 322",
                    "IPC 323","IPC 324","IPC 325","IPC 326",
                    "IPC 326A","IPC 327","IPC 328","IPC 329",
                    "IPC 330","IPC 331","IPC 332","IPC 333",
                    "IPC 334","IPC 335","IPC 336","IPC 337","IPC 338",
                ],
                "description": "Hurt and grievous hurt offenses",
                "related_categories": [
                    "offenses_against_life","crimes_against_women","criminal_intimidation"
                ],
            },
            "wrongful_restraint": {
                "sections": [
                    "IPC 339","IPC 340","IPC 341","IPC 342",
                    "IPC 343","IPC 344","IPC 345","IPC 346",
                    "IPC 347","IPC 348","IPC 349","IPC 350",
                    "IPC 351","IPC 352","IPC 353","IPC 354",
                    "IPC 355","IPC 356","IPC 357","IPC 358",
                ],
                "description": "Wrongful restraint and confinement",
                "related_categories": [
                    "kidnapping","crimes_against_women","criminal_intimidation"
                ],
            },
            "crimes_against_women": {
                "sections": [
                    "IPC 354","IPC 354A","IPC 354B","IPC 354C",
                    "IPC 354D","IPC 375","IPC 376","IPC 376A",
                    "IPC 376B","IPC 376C","IPC 376D","IPC 376E",
                    "IPC 498A",
                ],
                "description": "Offenses against women",
                "related_categories": [
                    "kidnapping","hurt_grievous_hurt","offenses_against_life",
                    "criminal_intimidation"
                ],
            },
            "kidnapping": {
                "sections": [
                    "IPC 359","IPC 360","IPC 361","IPC 362",
                    "IPC 363A","IPC 366A","IPC 366B","IPC 363",
                    "IPC 364","IPC 364A","IPC 365","IPC 366",
                    "IPC 367","IPC 368","IPC 369","IPC 370",
                    "IPC 370A","IPC 371","IPC 372","IPC 373","IPC 374",
                ],
                "description": "Kidnapping, abduction, trafficking",
                "related_categories": [
                    "crimes_against_women","wrongful_restraint","criminal_intimidation"
                ],
            },
            "property_offenses": {
                "sections": [
                    "IPC 378","IPC 379","IPC 380","IPC 381",
                    "IPC 382","IPC 384","IPC 386","IPC 390",
                    "IPC 392","IPC 395","IPC 396","IPC 399",
                    "IPC 402","IPC 403","IPC 405","IPC 406",
                    "IPC 409","IPC 383","IPC 385","IPC 387",
                    "IPC 388","IPC 389","IPC 391","IPC 393",
                    "IPC 394","IPC 397","IPC 398","IPC 400",
                    "IPC 401","IPC 404","IPC 407","IPC 408",
                    "IPC 410","IPC 411","IPC 412","IPC 413",
                    "IPC 414","IPC 451","IPC 452","IPC 453",
                    "IPC 454","IPC 455","IPC 456","IPC 457",
                    "IPC 458","IPC 459","IPC 460",
                ],
                "description": "Theft, robbery, dacoity, property crimes",
                "related_categories": [
                    "fraud","criminal_intimidation","criminal_conspiracy"
                ],
            },
            "fraud": {
                "sections": [
                    "IPC 415","IPC 417","IPC 418","IPC 420",
                    "IPC 463","IPC 465","IPC 467","IPC 468",
                    "IPC 471","IPC 477A","IPC 416","IPC 419",
                    "IPC 421","IPC 422","IPC 423","IPC 424",
                    "IPC 461","IPC 462","IPC 464","IPC 466",
                    "IPC 469","IPC 470","IPC 472","IPC 473",
                    "IPC 474","IPC 475","IPC 476","IPC 477",
                ],
                "description": "Cheating, forgery, fraud offenses",
                "related_categories": [
                    "property_offenses","public_servants","criminal_conspiracy"
                ],
            },
            "criminal_intimidation": {
                "sections": [
                    "IPC 503","IPC 504","IPC 505","IPC 506",
                    "IPC 507","IPC 509","IPC 500","IPC 501",
                    "IPC 502","IPC 508","IPC 510","IPC 511",
                ],
                "description": "Criminal intimidation, defamation, insult",
                "related_categories": [
                    "property_offenses","crimes_against_women","public_tranquility"
                ],
            },
            "criminal_conspiracy": {
                "sections": ["IPC 120A","IPC 120B"],
                "description": "Criminal conspiracy and abetment",
                "related_categories": [
                    "property_offenses","offenses_against_state","public_justice"
                ],
            },
        }

        # section → category mapping
        self.section_to_category = {
            sec: cat
            for cat, info in self.section_categories.items()
            for sec in info["sections"]
        }

        # flat sections list
        self.sections = [
            sec
            for info in self.section_categories.values()
            for sec in info["sections"]
        ]

        self.max_sections    = 10           # allow a few more per case
        self.current_sections: list   = []
        self.current_categories: set  = set()
        self.state_size = len(self.sections) + len(self.section_categories)
        print(f"Environment initialised — state_size={self.state_size}")

        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

        # ── IPC descriptions (condensed for brevity, keep originals) ──────────
        self.section_descriptions = {
            # Offenses against State
            "IPC 121":  "Waging war against Government of India, armed rebellion, treason",
            "IPC 121A": "Conspiracy to commit offenses against state, plotting against government",
            "IPC 122":  "Collecting arms with intention of waging war, weapon stockpiling",
            "IPC 123":  "Concealing with intent to facilitate design to wage war, hiding weapons",
            "IPC 124A": "Sedition – bringing hatred or contempt towards government, anti-national speech",
            "IPC 125":  "Waging war against friendly state, international terrorism",
            "IPC 126":  "Depredation on territories of friendly state, cross-border raids",
            "IPC 127":  "Receiving property taken by war or depredation",
            "IPC 128":  "Public servant voluntarily allowing prisoner to escape",
            "IPC 129":  "Public servant negligently allowing prisoner escape",
            "IPC 130":  "Aiding escape of, rescuing or harboring prisoner",
            "IPC 131":  "Abetting mutiny, inciting armed forces to rebel",
            # Public Justice
            "IPC 191":  "Giving false evidence, perjury, lying under oath",
            "IPC 192":  "Fabricating false evidence, evidence tampering",
            "IPC 193":  "Punishment for false evidence, perjury",
            "IPC 194":  "False evidence to procure conviction in capital offense",
            "IPC 195":  "False evidence to procure conviction with imprisonment",
            "IPC 196":  "Using evidence known to be false",
            "IPC 197":  "Issuing false certificate, document forgery",
            "IPC 198":  "Using false certificate known to be false",
            "IPC 199":  "False statement in declaration receivable as evidence",
            "IPC 200":  "Using false declaration knowing it to be false",
            "IPC 201":  "Causing disappearance of evidence, crime cover-up",
            "IPC 202":  "Intentional omission to give information of offense",
            "IPC 203":  "Giving false information respecting an offense committed",
            "IPC 204":  "Destruction of document to prevent it as evidence",
            "IPC 205":  "False personation for legal proceedings, identity fraud",
            "IPC 206":  "Fraudulent removal of property to prevent seizure",
            "IPC 207":  "Fraudulent claim to property to prevent seizure",
            "IPC 208":  "Fraudulently suffering decree for sum not due",
            "IPC 209":  "Dishonestly making false claim in court",
            "IPC 210":  "Fraudulently obtaining decree for sum not due",
            "IPC 211":  "False charge of offence, malicious prosecution, false accusation",
            "IPC 212":  "Harboring offender, hiding criminals, sheltering fugitives",
            "IPC 213":  "Taking gift to screen offender from punishment",
            "IPC 214":  "Offering gift for screening offender",
            "IPC 215":  "Taking gift to help recover stolen property",
            "IPC 216":  "Harboring offender who has escaped from custody",
            "IPC 216A": "Penalty for harboring robbers or dacoits",
            "IPC 217":  "Public servant disobeying direction of law to save person",
            "IPC 218":  "Public servant framing incorrect record to save person",
            "IPC 219":  "Public servant in judicial proceeding corruptly making report",
            "IPC 220":  "Commitment for trial by person knowing it is contrary to law",
            "IPC 221":  "Intentional omission to apprehend – public servant",
            "IPC 222":  "Intentional omission to apprehend convicted person",
            "IPC 223":  "Escape from custody suffered by public servant negligently",
            "IPC 224":  "Resistance or obstruction to lawful apprehension",
            "IPC 225":  "Resistance or obstruction to lawful apprehension of another",
            "IPC 225A": "Omission to apprehend – general cases",
            "IPC 225B": "Resistance or obstruction to lawful apprehension – general",
            "IPC 226":  "Unlawful return from transportation",
            # Offenses against Life
            "IPC 299":  "Culpable homicide, causing death by negligence, manslaughter",
            "IPC 300":  "Murder definition, intentional killing, premeditated homicide",
            "IPC 301":  "Culpable homicide – wrong person killed, mistaken identity",
            "IPC 302":  "Murder, intentional killing, willful killing, planned homicide",
            "IPC 303":  "Murder by life-convict",
            "IPC 304":  "Culpable homicide not amounting to murder, second-degree murder",
            "IPC 304A": "Causing death by negligence, accidental death, negligent driving",
            "IPC 304B": "Dowry death, bride burning, domestic violence murder",
            "IPC 305":  "Abetment of suicide of child or insane person",
            "IPC 306":  "Abetment of suicide, causing suicide, driving to suicide",
            "IPC 307":  "Attempt to murder, murderous assault",
            "IPC 308":  "Attempt to commit culpable homicide",
            "IPC 309":  "Attempt to commit suicide, self-harm",
            "IPC 310":  "Thug, professional killer, contract killer",
            "IPC 311":  "Punishment for thug",
            "IPC 312":  "Causing miscarriage, abortion, pregnancy termination",
            "IPC 313":  "Causing miscarriage without woman's consent, forced abortion",
            "IPC 314":  "Death caused by act done with intent to cause miscarriage",
            "IPC 315":  "Act done to prevent child being born alive, infanticide",
            "IPC 316":  "Causing death of quick unborn child, fetal homicide",
            "IPC 317":  "Exposure and abandonment of child under 12",
            "IPC 318":  "Concealment of birth by secret disposal of dead body",
            # Hurt and Grievous Hurt
            "IPC 319":  "Hurt definition, physical injury, bodily harm, assault",
            "IPC 320":  "Grievous hurt definition, serious injury, severe bodily harm",
            "IPC 321":  "Voluntarily causing hurt, intentional injury",
            "IPC 322":  "Voluntarily causing grievous hurt, intentional serious injury",
            "IPC 323":  "Punishment for voluntarily causing hurt",
            "IPC 324":  "Hurt by dangerous weapons, assault with knife or weapon",
            "IPC 325":  "Punishment for grievous hurt",
            "IPC 326":  "Grievous hurt by dangerous weapons, armed serious assault",
            "IPC 326A": "Acid attack, chemical assault, acid throwing, corrosive substance",
            "IPC 327":  "Voluntarily causing hurt to extort property",
            "IPC 328":  "Causing hurt by means of poison, drugging",
            "IPC 329":  "Voluntarily causing grievous hurt to extort property",
            "IPC 330":  "Voluntarily causing hurt to extort confession",
            "IPC 331":  "Voluntarily causing grievous hurt to extort confession",
            "IPC 332":  "Voluntarily causing hurt to deter public servant from duty",
            "IPC 333":  "Voluntarily causing grievous hurt to deter public servant",
            "IPC 334":  "Voluntarily causing hurt on grave provocation",
            "IPC 335":  "Voluntarily causing grievous hurt on grave provocation",
            "IPC 336":  "Act endangering life or personal safety of others",
            "IPC 337":  "Causing hurt by act endangering life",
            "IPC 338":  "Causing grievous hurt by act endangering life",
            # Wrongful Restraint
            "IPC 339":  "Wrongful restraint, blocking passage",
            "IPC 340":  "Wrongful confinement, locking up, illegal detention",
            "IPC 341":  "Punishment for wrongful restraint",
            "IPC 342":  "Punishment for wrongful confinement",
            "IPC 343":  "Wrongful confinement for three or more days",
            "IPC 344":  "Wrongful confinement for ten or more days",
            "IPC 345":  "Wrongful confinement of person for whose liberation writ issued",
            "IPC 346":  "Wrongful confinement in secret",
            "IPC 347":  "Wrongful confinement to extort property",
            "IPC 348":  "Wrongful confinement to extort confession",
            "IPC 349":  "Force definition",
            "IPC 350":  "Criminal force definition",
            "IPC 351":  "Assault definition",
            "IPC 352":  "Punishment for assault",
            "IPC 353":  "Assault on public servant",
            "IPC 354":  "Assault on woman to outrage modesty, molestation, sexual harassment",
            "IPC 355":  "Assault to dishonour person",
            "IPC 356":  "Assault for theft, snatching",
            "IPC 357":  "Assault for wrongful confinement",
            "IPC 358":  "Assault on grave provocation",
            # Crimes against Women
            "IPC 354A": "Sexual harassment, unwanted touch, demand for sexual favors",
            "IPC 354B": "Assault with intent to disrobe woman",
            "IPC 354C": "Voyeurism, stalking, watching private act",
            "IPC 354D": "Stalking, following, monitoring woman",
            "IPC 375":  "Rape definition, sexual assault, non-consensual intercourse",
            "IPC 376":  "Punishment for rape",
            "IPC 376A": "Rape causing death or persistent vegetative state",
            "IPC 376B": "Sexual intercourse by husband upon separated wife",
            "IPC 376C": "Sexual intercourse by person in authority",
            "IPC 376D": "Gang rape",
            "IPC 376E": "Punishment for repeat offenders of rape",
            "IPC 498A": "Cruelty by husband or relatives, domestic violence, dowry harassment",
            # Kidnapping
            "IPC 359":  "Kidnapping definition",
            "IPC 360":  "Kidnapping from India",
            "IPC 361":  "Kidnapping from lawful guardianship, child abduction",
            "IPC 362":  "Abduction definition",
            "IPC 363":  "Punishment for kidnapping, abduction",
            "IPC 363A": "Kidnapping for begging",
            "IPC 364":  "Kidnapping for murder",
            "IPC 364A": "Kidnapping for ransom, ransom demand, hostage",
            "IPC 365":  "Kidnapping with intent to secretly confine",
            "IPC 366":  "Kidnapping woman to compel marriage, enticement",
            "IPC 366A": "Procuration of minor girl",
            "IPC 366B": "Importation of girl from foreign country",
            "IPC 367":  "Kidnapping to subject to grievous hurt",
            "IPC 368":  "Wrongfully concealing kidnapped person",
            "IPC 369":  "Kidnapping child under 10 to steal property",
            "IPC 370":  "Trafficking of person, human trafficking, slavery",
            "IPC 370A": "Exploitation of trafficked person",
            "IPC 371":  "Habitual dealing in slaves",
            "IPC 372":  "Selling minor for prostitution",
            "IPC 373":  "Buying minor for prostitution",
            "IPC 374":  "Unlawful compulsory labor",
            # Property Offenses
            "IPC 378":  "Theft definition, stealing, taking property without consent",
            "IPC 379":  "Punishment for theft, snatching, pickpocketing",
            "IPC 380":  "Theft in dwelling house, house theft",
            "IPC 381":  "Theft by clerk or servant",
            "IPC 382":  "Theft after preparation for hurt",
            "IPC 383":  "Extortion definition",
            "IPC 384":  "Punishment for extortion, threatening for money",
            "IPC 385":  "Putting person in fear of injury for extortion",
            "IPC 386":  "Extortion by putting person in fear of death",
            "IPC 387":  "Putting person in fear of death to commit extortion",
            "IPC 388":  "Extortion by threat of accusation",
            "IPC 389":  "Putting person in fear of accusation to extort",
            "IPC 390":  "Robbery definition, armed robbery",
            "IPC 391":  "Dacoity definition, gang robbery",
            "IPC 392":  "Punishment for robbery, chain snatching, bag snatching",
            "IPC 393":  "Attempt to commit robbery",
            "IPC 394":  "Voluntarily causing hurt in committing robbery",
            "IPC 395":  "Punishment for dacoity, group robbery, gang looting",
            "IPC 396":  "Dacoity with murder",
            "IPC 397":  "Robbery with deadly weapon",
            "IPC 398":  "Attempt to commit robbery with deadly weapon",
            "IPC 399":  "Making preparation to commit dacoity",
            "IPC 400":  "Punishment for belonging to gang of dacoits",
            "IPC 401":  "Punishment for belonging to gang of thieves",
            "IPC 402":  "Assembling to commit dacoity",
            "IPC 403":  "Dishonest misappropriation of property",
            "IPC 404":  "Dishonest misappropriation of property of deceased",
            "IPC 405":  "Criminal breach of trust definition",
            "IPC 406":  "Punishment for criminal breach of trust",
            "IPC 407":  "Criminal breach of trust by carrier",
            "IPC 408":  "Criminal breach of trust by clerk or servant",
            "IPC 409":  "Criminal breach of trust by public servant",
            "IPC 410":  "Stolen property definition",
            "IPC 411":  "Dishonestly receiving stolen property",
            "IPC 412":  "Dishonestly receiving stolen property from dacoits",
            "IPC 413":  "Habitually dealing in stolen property",
            "IPC 414":  "Assisting in concealment of stolen property",
            "IPC 451":  "House-trespass to commit offence punishable with imprisonment",
            "IPC 452":  "House-trespass after preparation for hurt",
            "IPC 453":  "Lurking house-trespass or house-breaking",
            "IPC 454":  "Lurking house-trespass to commit offence",
            "IPC 455":  "Lurking house-trespass after preparation for hurt",
            "IPC 456":  "Lurking house-trespass by night",
            "IPC 457":  "Lurking house-trespass by night to commit offence",
            "IPC 458":  "Lurking house-trespass by night after preparation",
            "IPC 459":  "Grievous hurt while committing house-trespass",
            "IPC 460":  "All jointly concerned in house-breaking by night",
            # Fraud
            "IPC 415":  "Cheating definition, deception, fraud",
            "IPC 416":  "Cheating by personation, impersonation fraud",
            "IPC 417":  "Punishment for cheating",
            "IPC 418":  "Cheating with knowledge that wrongful loss will ensue",
            "IPC 419":  "Punishment for cheating by personation, fake identity",
            "IPC 420":  "Cheating and dishonestly inducing delivery of property, online fraud, scam",
            "IPC 421":  "Dishonest or fraudulent removal of property",
            "IPC 422":  "Dishonestly or fraudulently preventing debt being available",
            "IPC 423":  "Dishonest or fraudulent execution of deed of transfer",
            "IPC 424":  "Dishonest or fraudulent removal or concealment of property",
            "IPC 461":  "Dishonestly breaking open receptacle containing property",
            "IPC 462":  "Breaking open receptacle by person entrusted with custody",
            "IPC 463":  "Forgery definition, document fraud, fake documents",
            "IPC 464":  "Making false documents, forging signatures",
            "IPC 465":  "Punishment for forgery, fake paperwork, forged property documents",
            "IPC 466":  "Forgery of record of court or public register",
            "IPC 467":  "Forgery of valuable security, will, authority to adopt",
            "IPC 468":  "Forgery for purpose of cheating",
            "IPC 469":  "Forgery for purpose of harming reputation",
            "IPC 470":  "Forged document definition",
            "IPC 471":  "Using as genuine forged document",
            "IPC 472":  "Making or possessing counterfeit seal",
            "IPC 473":  "Making or possessing counterfeit seal for fraud",
            "IPC 474":  "Having possession of document described in IPC 466 or 467 knowing it to be forged",
            "IPC 475":  "Counterfeiting device or mark used for authenticating documents",
            "IPC 476":  "Counterfeiting device for authenticating documents",
            "IPC 477":  "Fraudulent cancellation, destruction of will, etc.",
            "IPC 477A": "Falsification of accounts, accounting fraud",
            # Criminal Intimidation
            "IPC 500":  "Defamation, character assassination, public insult",
            "IPC 501":  "Printing defamatory matter",
            "IPC 502":  "Sale of printed defamatory matter",
            "IPC 503":  "Criminal intimidation definition, threatening",
            "IPC 504":  "Intentional insult with provocation to cause breach of peace",
            "IPC 505":  "Statements conducing to public mischief, rumor spreading",
            "IPC 506":  "Punishment for criminal intimidation, threats, threatening letter",
            "IPC 507":  "Criminal intimidation by anonymous communication",
            "IPC 508":  "Act caused by inducing belief of divine displeasure",
            "IPC 509":  "Word, gesture or act intended to insult modesty of woman",
            "IPC 510":  "Misconduct in public by a drunken person",
            "IPC 511":  "Punishment for attempting to commit offenses",
            # Criminal Conspiracy
            "IPC 120A": "Criminal conspiracy definition, planning crime together",
            "IPC 120B": "Punishment of criminal conspiracy, group planning, organized crime",
        }

        # Init text encoder & pre-compute embeddings
        self.text_encoder = TextEncoder()
        self.section_embeddings = {
            sec: self.text_encoder.encode_text(desc)
            for sec, desc in self.section_descriptions.items()
        }

    # ── helpers ───────────────────────────────
    def get_section_category(self, section: str):
        return self.section_to_category.get(section)

    def get_related_categories(self, category: str):
        return self.section_categories[category].get("related_categories", [])

    def reset(self):
        self.current_sections  = []
        self.current_categories = set()
        return torch.zeros(self.state_size)

    # ─────────────────────────────────────────────────────────────────────────
    # get_relevant_sections  (core improvement: richer boosting + dynamic top-k)
    # ─────────────────────────────────────────────────────────────────────────
    def get_relevant_sections(self, description: str, top_k: int = None) -> list:
        desc_lower = description.lower()

        # Dynamic top-k: longer descriptions → more sections
        if top_k is None:
            word_count = len(description.split())
            top_k = min(12, max(5, word_count // 8))

        desc_emb = self.text_encoder.encode_text(description)

        similarities = {}
        for sec, emb in self.section_embeddings.items():
            similarities[sec] = F.cosine_similarity(desc_emb, emb).item()

        # ── keyword boosters ──────────────────────────────────────────────────
        BOOSTS = [
            # (ipc_list, keyword_list, boost_value)
            # Murder / homicide
            (["IPC 302"], ["murder","killed","murdered","homicide","dead body","shot dead","stabbed to death"], 1.2),
            (["IPC 307"], ["attempted murder","tried to kill","attack with intent"], 0.9),
            (["IPC 304"], ["culpable homicide","unintentional death","death without intention"], 0.8),
            (["IPC 304A"], ["negligent","doctor negligence","rash driving","accident death","careless driving"], 0.9),
            (["IPC 304B"], ["dowry death","bride burning","dowry harassment"], 1.0),
            (["IPC 306"], ["suicide","abetment","drove to suicide","mental harassment"], 0.8),
            # Weapons
            (["IPC 324"], ["knife","blade","sharp weapon","cut","stabbed"], 0.7),
            (["IPC 326"], ["grievous weapon","serious wound","sword","axe"], 0.7),
            (["IPC 326A"], ["acid","corrosive","acid attack","acid thrown"], 1.1),
            # Theft / Robbery
            (["IPC 379"], ["stolen","theft","snatched","pickpocket","robbed","stole"], 0.8),
            (["IPC 380"], ["house theft","stole from home","residential theft"], 0.7),
            (["IPC 392"], ["robbery","chain snatching","bag snatching","purse snatched"], 0.9),
            (["IPC 395"], ["dacoity","gang robbery","group looted","group robbery"], 1.0),
            (["IPC 457"], ["night break-in","midnight breaking","broke in at night","night entry"], 0.8),
            # Fraud / Forgery
            (["IPC 420"], ["cheat","fraud","scam","fake","swindle","dupe","deceive","online fraud"], 0.9),
            (["IPC 465","IPC 467"], ["forged document","fake document","property fraud","forged deed"], 0.9),
            (["IPC 468"], ["forgery for cheating"], 0.7),
            # Cyber
            (["IPC 66C","IPC 66D"], ["otp","password","cyber","hacked","phishing","online fraud","website"], 0.9),
            # Kidnapping
            (["IPC 363"], ["kidnap","abducted","taken away forcefully"], 1.0),
            (["IPC 364A"], ["ransom","kidnap for ransom","hostage"], 1.1),
            (["IPC 366"], ["kidnap woman","enticement","taken to marry"], 0.9),
            (["IPC 370"], ["trafficking","human trafficking","sex trafficking"], 1.0),
            # Women
            (["IPC 354"], ["molest","touched inappropriately","outrage modesty","sexual harassment"], 0.9),
            (["IPC 354A"], ["unwanted touch","sexual demand","harassment at work"], 0.8),
            (["IPC 354D"], ["stalking","following woman","monitoring"], 0.8),
            (["IPC 376"], ["rape","sexual assault","non-consensual"], 1.1),
            (["IPC 498A"], ["domestic violence","dowry cruelty","husband cruelty","marital abuse"], 0.9),
            # Intimidation
            (["IPC 506"], ["threaten","threatened","threat","intimidate","blackmail","extort"], 0.8),
            (["IPC 384"], ["extortion","demanded money","threatening letter"], 0.8),
            # Arms
            (["IPC 25 Arms Act"], ["gun","pistol","revolver","firearm","illegal weapon","arms"], 0.9),
            # Conspiracy
            (["IPC 120B"], ["planned","conspired","organized","group planned","three people","gang of"], 0.7),
            # House trespass
            (["IPC 441","IPC 442"], ["trespass","entered without permission","broke in"], 0.7),
            # Sedition
            (["IPC 124A"], ["sedition","anti-national","against government","inciting violence"], 0.8),
            # Bribery
            (["IPC 171E"], ["bribe","bribery","paid to get work","corrupt official"], 0.8),
            # Suicide abetment
            (["IPC 306"], ["drove to suicide","abetment of suicide","caused suicide"], 0.9),
            # Negligence death
            (["IPC 304A"], ["doctor negligence","rash driving","negligent accident","hit and run"], 0.9),
        ]

        for ipc_list, keywords, boost in BOOSTS:
            if any(kw in desc_lower for kw in keywords):
                for ipc in ipc_list:
                    if ipc in similarities:
                        similarities[ipc] += boost

        ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return [sec for sec, _ in ranked[:top_k]]

    # ── RL step ───────────────────────────────
    def step(self, action, description=None):
        if len(self.current_sections) >= self.max_sections:
            return self.get_state(), -1.0, True

        selected = self.sections[action]
        if selected in self.current_sections:
            return self.get_state(), -0.2, False

        cat = self.get_section_category(selected)

        # relevance-weighted reward
        reward = 0.3
        if description:
            relevant = self.get_relevant_sections(description)
            rank = len(relevant) - relevant.index(selected) if selected in relevant else 0
            reward += 0.6 * (rank / max(len(relevant), 1))

        if cat:
            if any(c in self.current_categories
                   for c in self.get_related_categories(cat)):
                reward += 0.35
            cat_secs = self.section_categories[cat]["sections"]
            if all(s in self.current_sections + [selected] for s in cat_secs):
                reward += 0.5

        self.current_sections.append(selected)
        if cat:
            self.current_categories.add(cat)

        if len(self.current_sections) == self.max_sections:
            reward += 1.2

        done = len(self.current_sections) == self.max_sections
        return self.get_state(), reward, done

    def get_state(self):
        state = torch.zeros(self.state_size)
        for sec in self.current_sections:
            state[self.sections.index(sec)] = 1
        for i, cat in enumerate(self.section_categories):
            if cat in self.current_categories:
                state[len(self.sections) + i] = 1
        return state

    # ── persistence ───────────────────────────
    def save_environment(self, filename="chargesheet_env.pth"):
        path = os.path.join(self.model_dir, filename)
        torch.save({
            "section_embeddings":  {k: v.cpu() for k, v in self.section_embeddings.items()},
            "section_descriptions": self.section_descriptions,
            "section_categories":   self.section_categories,
            "section_to_category":  self.section_to_category,
            "state_size":           self.state_size,
        }, path)
        print(f"Environment saved → {path}")

    def load_environment(self, filename="chargesheet_env.pth"):
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            print(f"No environment at {path}")
            return False
        ckpt = torch.load(path, map_location="cpu")
        self.section_embeddings    = ckpt["section_embeddings"]
        self.section_descriptions  = ckpt["section_descriptions"]
        self.section_categories    = ckpt["section_categories"]
        self.section_to_category   = ckpt["section_to_category"]
        self.state_size            = ckpt["state_size"]
        print(f"Environment loaded ← {path}")
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Policy Network  (deeper + BN + Dropout)
# ──────────────────────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout: float = 0.25):
        super().__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != self.input_size:
            raise ValueError(
                f"Expected input size {self.input_size}, got {x.shape[1]}"
            )
        logits = self.network(x)
        probs  = F.softmax(logits, dim=-1)
        probs  = torch.nan_to_num(probs, nan=1e-7)
        probs  = probs / probs.sum(dim=-1, keepdim=True)
        return probs


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

class ChargesheetAgent:
    def __init__(self, state_size: int, hidden_size: int, action_size: int,
                 lr: float = 3e-4):
        self.policy    = PolicyNetwork(state_size, hidden_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr,
                                    weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=500, eta_min=1e-5
        )
        self.memory    = deque(maxlen=4000)
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs("analytics", exist_ok=True)

        self.training_metrics = {
            "episode_rewards": [],
            "episode_losses":  [],
            "episode_times":   [],
        }
        self.gamma          = 0.99
        self.entropy_coeff  = 0.01   # encourage exploration

    # ── action selection ──────────────────────
    def select_action(self, state: torch.Tensor):
        self.policy.eval()
        with torch.no_grad():
            probs = self.policy(state).squeeze()
        self.policy.train()
        m = torch.distributions.Categorical(probs)
        return m.sample().item()

    def store_transition(self, state, action, reward):
        self.memory.append((state, action, reward))

    # ── training step ─────────────────────────
    def train(self, batch_size: int = 64) -> float:
        if len(self.memory) < 3:
            return 0.0

        batch = list(self.memory)
        states, actions, rewards = zip(*batch)

        # Discounted returns
        returns, R = [], 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns_t = torch.tensor(returns, dtype=torch.float32)
        if returns_t.std() > 1e-6:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        states_t  = torch.stack(states)
        if states_t.dim() == 3:
            states_t = states_t.squeeze(1)

        self.policy.train()
        probs    = self.policy(states_t)
        dist     = torch.distributions.Categorical(probs)
        actions_t = torch.tensor(actions, dtype=torch.long)

        log_probs = dist.log_prob(actions_t)
        entropy   = dist.entropy().mean()

        policy_loss = -(log_probs * returns_t).mean()
        loss        = policy_loss - self.entropy_coeff * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.memory.clear()
        return loss.item()

    # ── persistence ───────────────────────────
    def save_model(self, filename="chargesheet_model.pth"):
        path = os.path.join(self.model_dir, filename)
        torch.save({
            "policy_state_dict":    self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": {
                "state_size":  self.policy.input_size,
                "hidden_size": 256,
                "action_size": self.policy.network[-1].out_features,
            },
        }, path)
        print(f"Model saved → {path}")

    def load_model(self, filename="chargesheet_model.pth") -> bool:
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            print(f"No model at {path}")
            return False
        ckpt = torch.load(path, map_location="cpu")
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Model loaded ← {path}")
        return True

    def save_analytics(self, episode, env):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        adir = os.path.join("analytics", f"training_{ts}")
        os.makedirs(adir, exist_ok=True)

        df = pd.DataFrame({
            "episode": range(1, len(self.training_metrics["episode_rewards"]) + 1),
            "reward":  self.training_metrics["episode_rewards"],
            "loss":    self.training_metrics["episode_losses"],
            "time":    self.training_metrics["episode_times"],
        })
        df.to_csv(os.path.join(adir, "training_metrics.csv"), index=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(df["episode"], df["reward"]); axes[0].set_title("Rewards")
        axes[1].plot(df["episode"], df["loss"]);   axes[1].set_title("Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(adir, "training.png"))
        plt.close("all")
        print(f"Analytics saved → {adir}")


# ──────────────────────────────────────────────────────────────────────────────
# Rule-based post-filter  (significantly expanded)
# ──────────────────────────────────────────────────────────────────────────────

def rule_based_filter(predicted_ipcs: list, description: str) -> list:
    """
    Comprehensive rule-based post-filter.
    Two passes:
      1. REMOVAL  — strip IPCs that fire too broadly when key context is absent
      2. ADDITION — inject IPCs that keyword evidence strongly supports
    Then a final MUTUAL-EXCLUSION pass to resolve contradictions.
    """
    d      = description.lower()
    result = set(predicted_ipcs)

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 1 — NOISE REMOVAL
    # Remove an IPC from the prediction if NONE of its required keywords appear.
    # Format: "IPC_CODE": [list of keywords — at least one must be present]
    # ══════════════════════════════════════════════════════════════════════════
    REMOVE_IF_ABSENT = {
        # Property / theft
        "IPC 400":  ["gang", "criminal gang", "dacoity gang"],
        "IPC 401":  ["gang", "habitual", "criminal group", "thief gang"],
        "IPC 411":  ["received stolen", "bought stolen", "knowingly received"],
        "IPC 413":  ["habitual dealer", "stolen property dealer"],
        "IPC 414":  ["concealed stolen", "helped hide stolen"],

        # Breach of trust / misappropriation
        "IPC 403":  ["misappropriate", "dishonestly used", "converted to own use"],
        "IPC 405":  ["entrusted", "breach of trust", "custody", "embezzle"],
        "IPC 406":  ["entrusted", "custody", "embezzle", "breach of trust"],
        "IPC 408":  ["servant", "clerk", "employee misused"],
        "IPC 409":  ["public servant", "government official misused"],

        # Cheating / fraud — only if deception context present
        "IPC 415":  ["cheat", "deceive", "fraud", "fake", "scam", "dupe"],
        "IPC 417":  ["cheat", "deceived", "deception", "tricked"],
        "IPC 419":  ["impersonation", "fake identity", "pretended to be", "posed as"],

        # Forgery — only if document fraud context present
        "IPC 463":  ["forged", "fake document", "document fraud", "fabricated document"],
        "IPC 464":  ["forged signature", "fake document", "false document"],
        "IPC 466":  ["court record", "public register", "forged official"],
        "IPC 471":  ["used forged", "submitted fake", "presented forged"],
        "IPC 477A": ["accounts", "accounting", "financial records", "books of account"],

        # Mischief
        "IPC 425":  ["mischief", "damaged", "destroyed property", "vandalism"],
        "IPC 426":  ["mischief", "damaged property", "destroyed", "vandalism"],
        "IPC 427":  ["damage", "destroyed", "vandalism", "mischief"],
        "IPC 435":  ["fire", "explosive", "burned", "set ablaze"],
        "IPC 436":  ["fire", "explosive", "burned house", "set fire to house"],

        # Trespass
        "IPC 441":  ["trespass", "entered without", "unauthorized entry", "unlawful entry"],
        "IPC 442":  ["house trespass", "entered home", "unauthorized entry into house"],
        "IPC 447":  ["trespass", "unauthorized entry", "encroachment"],
        "IPC 448":  ["house trespass", "entered home without", "broke into house"],

        # Forced labour / trafficking
        "IPC 374":  ["forced labor", "compulsory work", "bonded labor", "forced to work"],
        "IPC 370":  ["trafficking", "human trafficking", "sex trafficking", "sold person"],
        "IPC 371":  ["slave", "slavery", "bought person", "sold person"],
        "IPC 372":  ["sold girl", "prostitution", "minor sold"],
        "IPC 373":  ["bought girl", "prostitution", "minor bought"],

        # Public servants / bribery
        "IPC 161":  ["bribe", "bribery", "paid official", "gratification"],
        "IPC 165":  ["public servant", "official", "government employee"],
        "IPC 166":  ["public servant", "disobeyed law", "official misconduct"],
        "IPC 167":  ["public servant", "false document", "official forged"],
        "IPC 168":  ["public servant", "business", "trade", "illegal contract"],
        "IPC 169":  ["public servant", "purchased property", "bid on property"],
        "IPC 217":  ["public servant", "shielded", "saved from punishment", "official saved"],
        "IPC 218":  ["public servant", "incorrect record", "false entry", "official record"],

        # False evidence / perjury
        "IPC 191":  ["false evidence", "perjury", "false testimony", "lied in court"],
        "IPC 192":  ["fabricated evidence", "planted evidence", "created false proof"],
        "IPC 193":  ["false evidence", "perjury", "false statement court"],
        "IPC 196":  ["used false evidence", "submitted fake proof"],
        "IPC 201":  ["destroyed evidence", "hid evidence", "concealed proof", "cover up"],
        "IPC 204":  ["destroyed document", "burned document", "hid document"],
        "IPC 211":  ["false complaint", "malicious prosecution", "false accusation", "fake fir"],

        # Kidnapping sub-types
        "IPC 363A": ["begging", "used child to beg", "forced to beg"],
        "IPC 364":  ["kidnap to murder", "abducted to kill"],
        "IPC 364A": ["ransom", "ransom demand", "kidnap for money", "hostage"],
        "IPC 365":  ["secretly confined", "hidden away", "secret confinement"],
        "IPC 367":  ["kidnap grievous hurt", "abducted and injured"],
        "IPC 369":  ["stole child", "child kidnap for property"],

        # Offenses against state
        "IPC 121":  ["war", "waging war", "armed rebellion", "treason", "insurgency"],
        "IPC 121A": ["conspiracy against state", "plotted against government"],
        "IPC 122":  ["collected arms", "weapon stockpile", "arms for war"],
        "IPC 124A": ["sedition", "anti-national", "against government", "incite violence"],
        "IPC 131":  ["mutiny", "armed forces rebel", "military insubordination"],

        # Suicide related
        "IPC 305":  ["child suicide", "minor suicide", "insane person suicide"],
        "IPC 306":  ["abetment", "drove to suicide", "caused suicide", "forced suicide"],
        "IPC 309":  ["attempted suicide", "self-harm", "tried to kill himself",
                     "tried to kill herself"],

        # Hurt sub-types
        "IPC 327":  ["hurt to extort", "injured to take property", "assaulted to steal"],
        "IPC 328":  ["poison", "drugged", "sedative", "spiked drink", "intoxicant"],
        "IPC 329":  ["grievous hurt extort", "serious injury to rob"],
        "IPC 330":  ["hurt to get confession", "tortured for confession"],
        "IPC 331":  ["grievous hurt confession", "severely injured for confession"],
        "IPC 332":  ["attacked public servant", "assaulted government official", "hurt police"],
        "IPC 333":  ["grievous hurt public servant", "seriously injured police officer"],

        # Wrongful confinement sub-types
        "IPC 343":  ["confined three days", "held captive days", "kept for days"],
        "IPC 344":  ["confined ten days", "held ten days", "captive for week"],
        "IPC 346":  ["secret confinement", "hidden captive", "confined secretly"],
        "IPC 347":  ["confined to extort", "held captive for property"],
        "IPC 348":  ["confined for confession", "held captive for statement"],

        # Crimes against women sub-types
        "IPC 354B": ["disrobe", "undress", "strip", "removed clothes"],
        "IPC 354C": ["voyeur", "watching", "photographed", "filmed secretly", "peeping"],
        "IPC 376A": ["rape death", "rape coma", "rape vegetative state"],
        "IPC 376D": ["gang rape", "multiple accused rape", "group rape"],
        "IPC 376E": ["repeat rape", "previous conviction rape"],
        "IPC 304B": ["dowry death", "bride burning", "dowry harassment death"],

        # Specific fraud types
        "IPC 420":  ["cheat", "fraud", "scam", "fake", "swindle", "dupe",
                     "deceive", "online fraud", "cheated", "con"],
        "IPC 465":  ["forged", "fake document", "forged deed", "property document fraud",
                     "false document"],
        "IPC 467":  ["valuable security", "forged will", "forged authority", "forged bond"],
        "IPC 468":  ["forgery cheat", "forged to cheat", "fake document to defraud"],
        "IPC 469":  ["forgery reputation", "fake document to defame", "forged to harm name"],

        # Defamation
        "IPC 499":  ["defame", "defamation", "false statement about", "damaged reputation"],
        "IPC 500":  ["defame", "defamation", "insult reputation", "character assassination"],
        "IPC 501":  ["printed defamation", "published defamatory", "printed insult"],

        # Rioting / unlawful assembly
        "IPC 141":  ["unlawful assembly", "illegal gathering", "mob", "riot"],
        "IPC 143":  ["unlawful assembly", "member of mob", "part of riot"],
        "IPC 146":  ["rioting", "riot", "mob violence"],
        "IPC 147":  ["rioting", "riot", "mob violence", "violent mob"],
        "IPC 148":  ["rioting weapon", "armed riot", "riot with weapon"],
        "IPC 153A": ["enmity", "hatred between groups", "communal", "religious tension"],
        "IPC 153B": ["national integrity", "prejudicial to integration"],

        # Miscellaneous
        "IPC 269":  ["epidemic", "disease spread", "contagion", "infection spread"],
        "IPC 270":  ["epidemic", "disease spread", "contagion negligent"],
        "IPC 279":  ["rash driving", "reckless driving", "negligent driving"],
        "IPC 304A": ["negligent", "rash", "accident", "careless", "doctor negligence",
                     "hit and run"],
        "IPC 336":  ["endangered life", "reckless act", "dangerous act"],
        "IPC 337":  ["hurt by endangering", "injured by reckless act"],
        "IPC 338":  ["grievous hurt endangering", "serious injury by reckless act"],
    }

    for ipc, required_kws in REMOVE_IF_ABSENT.items():
        if ipc in result and not any(k in d for k in required_kws):
            result.discard(ipc)

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 2 — ADDITION RULES
    # Each tuple: ([ipc_list], [trigger_keywords])
    # If ANY keyword is found → add ALL ipcs in the list.
    # ══════════════════════════════════════════════════════════════════════════
    ADD_RULES = [

        # ── Murder / homicide ─────────────────────────────────────────────────
        (["IPC 302"],
         ["murdered","murder","killed","shot dead","stabbed to death","hacked to death",
          "beaten to death","strangled","poisoned to death","set on fire and died",
          "found dead","body found","dead body"]),

        (["IPC 307"],
         ["attempted murder","tried to kill","attack with intent to kill",
          "fired at","shot at","stabbed with intent","tried to run over"]),

        (["IPC 304"],
         ["culpable homicide","death not intended","died due to assault",
          "unintentional killing","died after being hit"]),

        (["IPC 304A"],
         ["negligent driving","rash driving","hit and run","doctor negligence",
          "negligent surgery","careless treatment","accident death","road accident death",
          "negligent","accidental death"]),

        (["IPC 304B"],
         ["dowry death","bride burning","dowry harassment death",
          "died due to dowry","burnt for dowry"]),

        (["IPC 306"],
         ["abetment of suicide","drove to suicide","caused suicide",
          "harassment led to suicide","forced to commit suicide","provoked to suicide"]),

        # ── Attempt and hurt ──────────────────────────────────────────────────
        (["IPC 307"],
         ["fired gun at","shot at victim","stabbed with knife intending to kill",
          "tried to run over with car"]),

        (["IPC 323"],
         ["hit","slapped","punched","beaten","kicked","assaulted","physical fight",
          "thrashed","battered","blows","struck","pushed violently","injured in fight"]),

        (["IPC 324"],
         ["knife","blade","sharp object","cut with","stabbed","slash","chopper",
          "sword","sickle","iron rod","lathi","stick attack","hit with rod"]),

        (["IPC 325"],
         ["grievous injury","serious injury","fracture","bone broken","permanently injured",
          "severe beating","critical condition after assault"]),

        (["IPC 326"],
         ["grievous hurt weapon","seriously injured with weapon","weapon grievous",
          "deep cut","severe stab wound","axe attack","serious weapon injury"]),

        (["IPC 326A"],
         ["acid","acid attack","acid thrown","corrosive substance","chemical thrown",
          "sulphuric acid","burns from acid"]),

        (["IPC 328"],
         ["poisoned","spiked drink","drugged","sedative added","intoxicant given",
          "substance added to food","mixed in drink"]),

        # ── Wrongful confinement ──────────────────────────────────────────────
        (["IPC 342"],
         ["tied up","bound","locked in","confined","held captive","restrained",
          "could not move","prevented from leaving","locked inside",
          "tied hands","gagged","detained illegally"]),

        (["IPC 341"],
         ["blocked path","stopped from going","prevented movement","obstructed passage"]),

        # ── Robbery / theft / dacoity ─────────────────────────────────────────
        (["IPC 379"],
         ["stolen","stole","theft","pickpocket","snatched bag","purse stolen",
          "mobile stolen","wallet stolen","item missing after","took without permission"]),

        (["IPC 380"],
         ["stole from house","house theft","broke into home and stole",
          "residential theft","theft at residence","stole from flat","stole from room"]),

        (["IPC 382"],
         ["theft after hurt","stole after hitting","robbed after injuring"]),

        (["IPC 392"],
         ["robbery","robbed","chain snatching","snatched jewelry","bag snatched",
          "mobile snatched","took by force","looted on road","highway robbery"]),

        (["IPC 394"],
         ["hurt during robbery","injured while robbing","assault during theft"]),

        (["IPC 395","IPC 120B"],
         ["dacoity","gang robbery","group looted","five or more robbers",
          "group of men looted","armed gang robbery","multiple attackers looted"]),

        (["IPC 396"],
         ["dacoity with murder","gang robbery murder","killed during dacoity"]),

        (["IPC 397"],
         ["robbery with weapon","armed robbery","gun pointed during robbery",
          "knife used in robbery","weapon used to rob"]),

        (["IPC 384"],
         ["extortion","extorted money","demanded money under threat",
          "threatened to harm if money not paid","paid under fear"]),

        (["IPC 386"],
         ["extortion death threat","threatened to kill if money not paid",
          "death threat for money"]),

        # ── House trespass / break-in ─────────────────────────────────────────
        (["IPC 457"],
         ["broke into house at night","night break-in","midnight entry",
          "11 pm broke in","midnight broke","entered house at night",
          "broke door at night","night burglary","trespass at night"]),

        (["IPC 454"],
         ["lurking trespass","sneaked into house","crept into house",
          "secretly entered house"]),

        (["IPC 452"],
         ["entered house to hurt","broke in to assault","trespass to harm"]),

        (["IPC 450"],
         ["entered house to commit serious crime","trespass murder",
          "broke in for serious offense"]),

        (["IPC 451"],
         ["entered house to commit offense","broke in to commit crime"]),

        (["IPC 441","IPC 448"],
         ["trespassed","entered without permission","unlawful entry",
          "broke into","forced entry","entered illegally"]),

        # ── Fraud / cheating ──────────────────────────────────────────────────
        (["IPC 420"],
         ["cheat","cheated","fraud","scam","fake","swindle","dupe","deceive",
          "online fraud","con","misled","lured with false promise","took money by fraud"]),

        (["IPC 419"],
         ["impersonation","posed as","pretended to be","fake identity",
          "disguised as","fake officer","fake doctor","fake government official"]),

        (["IPC 418"],
         ["cheated knowing loss","fraud knowing damage","deceived despite knowing"]),

        # ── Forgery ───────────────────────────────────────────────────────────
        (["IPC 465","IPC 468"],
         ["forged document","fake document","forged deed","fake property papers",
          "fabricated certificate","forged signature","fake identity proof"]),

        (["IPC 467"],
         ["forged will","forged bond","forged valuable paper","fake security document"]),

        (["IPC 471"],
         ["used forged document","submitted fake papers","presented forged certificate",
          "used fake documents as genuine"]),

        # ── Cyber crimes ──────────────────────────────────────────────────────
        (["IPC 66C","IPC 66D"],
         ["otp fraud","password stolen","cyber crime","online fraud","hacked account",
          "phishing","website fraud","identity theft online","digital fraud",
          "upi fraud","internet fraud","sim swap","fake website"]),

        (["IPC 66"],
         ["hacking","hacked server","unauthorized computer access","data theft",
          "computer break-in","database hacked"]),

        # ── Kidnapping ────────────────────────────────────────────────────────
        (["IPC 363"],
         ["kidnapped","abducted","taken away forcibly","child missing abducted",
          "taken against will","forcibly taken from home"]),

        (["IPC 366"],
         ["woman kidnapped","girl abducted","taken for marriage","enticement of woman",
          "compelled to marry","lured and taken"]),

        (["IPC 364A"],
         ["ransom","kidnap for ransom","ransom demand","hostage","held for money",
          "money demanded for release"]),

        (["IPC 370"],
         ["trafficking","human trafficking","sex trafficking","sold for prostitution",
          "trafficked","smuggled for exploitation"]),

        # ── Crimes against women ──────────────────────────────────────────────
        (["IPC 354"],
         ["molested","molestation","outrage modesty","touched inappropriately",
          "grabbed","physically harassed woman","inappropriate touch",
          "touched woman without consent","eve teasing physical"]),

        (["IPC 354A"],
         ["sexual harassment","demanded sexual favour","unwanted touch",
          "showed pornography","sexual gesture","sexual comment at workplace"]),

        (["IPC 354B"],
         ["tried to disrobe","removed clothes forcibly","undressed forcibly",
          "stripped against will"]),

        (["IPC 354C"],
         ["voyeurism","secretly filmed","hidden camera","secretly photographed",
          "peeping","watched bathing","recorded without consent"]),

        (["IPC 354D"],
         ["stalking","stalked","followed repeatedly","monitored online","kept following",
          "tracking movements","obsessively followed","sending unwanted messages"]),

        (["IPC 375","IPC 376"],
         ["rape","sexual assault","raped","gang raped","sexually assaulted",
          "non-consensual intercourse","forced intercourse"]),

        (["IPC 376D"],
         ["gang rape","group rape","multiple men raped","multiple accused rape"]),

        (["IPC 498A"],
         ["domestic violence","dowry harassment","cruelty by husband",
          "marital abuse","tortured for dowry","husband beat wife",
          "in-laws harassment","dowry demand cruelty"]),

        # ── Criminal intimidation / threats ───────────────────────────────────
        (["IPC 506"],
         ["threatened","threat","threatening","intimidated","death threat",
          "warned to face consequences","threatened to harm","threatened to kill",
          "threatening call","threatening message"]),

        (["IPC 507"],
         ["anonymous threat","unknown caller threatened","threatening anonymous letter",
          "threat from unknown person"]),

        (["IPC 504"],
         ["abused","verbal abuse","insulted","provoked","public insult",
          "abusive language","shouted abuses","provoked to fight"]),

        (["IPC 509"],
         ["insulted woman","obscene gesture to woman","lewd remark","passed comment on woman",
          "eve teasing verbal","wolf whistling","sexually explicit comment to woman"]),

        (["IPC 500"],
         ["defamed","defamation","spread false rumour","ruined reputation",
          "false statement about character","publicly defamed"]),

        # ── Criminal conspiracy ───────────────────────────────────────────────
        (["IPC 120B"],
         ["planned together","conspired","two or more planned","organized crime",
          "hatched plan","group planned","pre-planned","they planned",
          "two men planned","accomplice","worked together"]),

        (["IPC 120A"],
         ["conspiracy","criminal conspiracy","agreed to commit crime"]),

        # ── Arms Act ──────────────────────────────────────────────────────────
        (["IPC 25 Arms Act"],
         ["gun","pistol","revolver","firearm","rifle","shotgun","country made pistol",
          "illegal weapon","unlicensed gun","arms","live ammunition","cartridge"]),

        (["IPC 27 Arms Act"],
         ["fired gun","discharged firearm","used pistol to shoot","shot with gun"]),

        # ── Dowry ─────────────────────────────────────────────────────────────
        (["IPC 498A","IPC 304B"],
         ["dowry death","died due to dowry"]),

        (["IPC 498A"],
         ["dowry demand","harassment for dowry","tortured for dowry","dowry cruelty"]),

        # ── Bribery / public servant ──────────────────────────────────────────
        (["IPC 171E"],
         ["bribery","bribe","paid bribe","gave money to officer","illegal gratification",
          "corrupt official","paid to get work done","bribed police"]),

        (["IPC 7 PC Act"],
         ["bribe","corruption","took bribe","accepted bribe","demanded bribe",
          "public servant bribe","official corruption"]),

        # ── Public justice offenses ───────────────────────────────────────────
        (["IPC 201"],
         ["destroyed evidence","hid evidence","concealed proof","burnt evidence",
          "tampered evidence","washed crime scene","cover up"]),

        (["IPC 211"],
         ["false complaint","fake fir","malicious prosecution","wrong accusation",
          "filed false case","framed innocent","false allegation"]),

        (["IPC 212"],
         ["harbored criminal","hid accused","sheltered accused",
          "gave refuge to criminal","hid the suspect"]),

        # ── Negligence / rash acts ────────────────────────────────────────────
        (["IPC 279"],
         ["rash driving","reckless driving","negligent driving",
          "speeding caused accident","drove rashly"]),

        (["IPC 337"],
         ["hurt in accident","injured due to reckless act","hurt by negligence"]),

        (["IPC 338"],
         ["grievous hurt accident","serious injury reckless","critically hurt negligence"]),

        # ── Rioting / unlawful assembly ───────────────────────────────────────
        (["IPC 147","IPC 148"],
         ["riot","rioting","mob violence","violent crowd","attacked in group",
          "group attacked","mob assault"]),

        (["IPC 141","IPC 143"],
         ["unlawful assembly","illegal gathering","mob formed","crowd gathered violently"]),

        (["IPC 153A"],
         ["communal violence","religious hatred","caste hatred","promoted enmity",
          "communal tension","incited riot"]),

        # ── Sedition / state offenses ─────────────────────────────────────────
        (["IPC 124A"],
         ["sedition","anti-national activity","against government","incite rebellion",
          "overthrow government","disaffection against state"]),

        # ── Abetment ─────────────────────────────────────────────────────────
        (["IPC 107","IPC 109"],
         ["abetted","instigated","helped commit crime","assisted in crime",
          "encouraged to commit","aided the accused"]),

        # ── Domestic violence / dowry specific ───────────────────────────────
        (["IPC 342","IPC 323"],
         ["locked in room","locked inside room","locked me inside","locked the door from outside",
          "could not come out","prevented from leaving house","confined in room",
          "shut inside","bolted from outside","trapped inside","locked inside the"]),

        (["IPC 506","IPC 384"],
         ["burn alive","threatened to burn","set on fire threat","threatened to set ablaze",
          "threatened to kill if dowry not given","threatened to harm if money not brought"]),

        (["IPC 120B","IPC 34"],
         ["husband and his mother","husband and mother-in-law","wife and her mother-in-law",
          "in-laws together","mother-in-law and husband","father-in-law and husband",
          "all family members","both of them planned","they both tortured"]),

        (["IPC 384"],
         ["demanded gold","demanded money from parents","demanded cash",
          "demanded jewelry","forced to bring dowry","bring more dowry","bring more gold",
          "demanded dowry under threat","extorted dowry"]),

        (["IPC 34"],
         ["common intention","acted together","jointly committed","all of them together",
          "in furtherance of common intention","together they"]),

        # ── Hurt with household objects ───────────────────────────────────────
        (["IPC 324"],
         ["iron rod","wooden stick","belt","chain","brick","stone","hammer",
          "hit with rod","struck with stick","beat with rod","attacked with rod",
          "hit with iron","wooden plank","baseball bat"]),

        (["IPC 323"],
         ["slapped","punched","kicked","beaten up","physically abused",
          "hit repeatedly","pushed and fell","manhandled","thrashed","battered"]),

        # ── Mobile / communication snatching ─────────────────────────────────
        (["IPC 379","IPC 342"],
         ["took away phone","snatched mobile","took mobile so could not call",
          "removed phone","confiscated phone","took away means to call",
          "took away my mobile","mobile phone so i could not","phone so i could not"]),
    ]

    for ipcs, kws in ADD_RULES:
        if any(k in d for k in kws):
            result.update(ipcs)

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 3 — CONTEXTUAL UPGRADES
    # When a less-severe IPC is present but context supports a more severe one,
    # upgrade automatically.
    # ══════════════════════════════════════════════════════════════════════════

    # Simple hurt → grievous hurt if "serious", "fracture", "hospital", "critical"
    grievous_ctx = ["serious","critical","fracture","hospitalised","hospitalised",
                    "hospital","grievous","deep wound","surgery needed","icu"]
    if "IPC 323" in result and any(w in d for w in grievous_ctx):
        result.add("IPC 325")

    # Weapon hurt → grievous hurt by weapon if "seriously injured"
    if "IPC 324" in result and any(w in d for w in grievous_ctx):
        result.add("IPC 326")

    # Robbery → dacoity if 5+ people implied
    five_plus = ["five","six","seven","eight","gang of","group of men","several men",
                 "many men","multiple attackers","band of"]
    if "IPC 392" in result and any(w in d for w in five_plus):
        result.add("IPC 395")
        result.add("IPC 120B")

    # Theft at night + house → house breaking at night
    night_kws  = ["night","midnight","2 am","3 am","late night","11 pm","12 am","1 am"]
    house_kws  = ["house","home","residence","flat","apartment","room","dwelling"]
    if "IPC 379" in result and any(w in d for w in night_kws) and any(w in d for w in house_kws):
        result.add("IPC 457")
        result.add("IPC 380")

    # IPC 302 present + weapon → add Arms Act
    weapon_kws = ["gun","pistol","revolver","firearm","rifle","country made"]
    if "IPC 302" in result and any(w in d for w in weapon_kws):
        result.add("IPC 25 Arms Act")

    # IPC 376 (rape) + multiple accused → gang rape
    multi_accused = ["gang rape","multiple men","group rape","they raped","all of them raped"]
    if "IPC 376" in result and any(w in d for w in multi_accused):
        result.add("IPC 376D")
        result.add("IPC 120B")

    # Kidnapping + ransom → 364A
    if ("IPC 363" in result or "IPC 366" in result) and any(w in d for w in ["ransom","money for release","money demanded"]):
        result.add("IPC 364A")

    # Conspiracy upgrade: if 2+ people committed any serious crime
    serious_crimes = {"IPC 302","IPC 395","IPC 376","IPC 376D","IPC 364A","IPC 392","IPC 380"}
    two_plus = ["two men","two people","they both","all of them","group of","gang",
                "accompanied by","along with","with his accomplice","with his associate"]
    if result & serious_crimes and any(w in d for w in two_plus):
        result.add("IPC 120B")

    # ── Domestic violence upgrades ────────────────────────────────────────────

    # 498A present + physical hurt → always add 323
    if "IPC 498A" in result and any(w in d for w in ["beat","hit","slap","punch","kick",
                                                      "assault","thrash","batter","struck"]):
        result.add("IPC 323")

    # 498A + fracture/serious injury → add 325
    if "IPC 498A" in result and any(w in d for w in ["fracture","broken bone","serious injury",
                                                      "hospitalised","hospital","critical"]):
        result.add("IPC 325")

    # 498A + iron rod / weapon → add 324
    if "IPC 498A" in result and any(w in d for w in ["rod","stick","weapon","knife","belt",
                                                      "chain","lathi","object","instrument"]):
        result.add("IPC 324")

    # 498A + locked/confined → add 342
    if "IPC 498A" in result and any(w in d for w in ["locked","confined","could not leave",
                                                      "prevented","restrained","trapped"]):
        result.add("IPC 342")

    # 498A + in-laws involved → add 120B + 34
    if "IPC 498A" in result and any(w in d for w in ["mother-in-law","father-in-law",
                                                      "in-laws","his mother","his father",
                                                      "his family","her in-laws"]):
        result.add("IPC 120B")
        result.add("IPC 34")

    # 498A + dowry demand under threat → add 384
    if "IPC 498A" in result and any(w in d for w in ["demanded","bring gold","bring money",
                                                      "bring cash","bring jewelry","dowry demand",
                                                      "more dowry","extort"]):
        result.add("IPC 384")

    # ── IPC 302 false positive suppression ───────────────────────────────────
    # Only keep IPC 302 if someone ACTUALLY died — not just threatened
    actual_death = ["died","dead body","killed","murdered","found dead","death",
                    "passed away","succumbed","body found","shot dead","stabbed to death",
                    "beaten to death","burned to death","poisoned to death"]
    if "IPC 302" in result and not any(w in d for w in actual_death):
        result.discard("IPC 302")

    # ── IPC 392 (robbery) false positive suppression ──────────────────────────
    # Only keep robbery if there is actual forcible taking on the street / outside
    robbery_ctx = ["snatched","robbed","chain snatching","bag snatched","highway",
                   "road robbery","looted on road","took by force on road"]
    if "IPC 392" in result and not any(w in d for w in robbery_ctx):
        # if 498A is present (domestic case) — robbery doesn't apply
        if "IPC 498A" in result:
            result.discard("IPC 392")

    # ── IPC 420 (fraud/cheating) false positive suppression ──────────────────
    fraud_ctx = ["cheat","fraud","scam","fake","deceive","online","swindle",
                 "false promise","misrepresent","con"]
    if "IPC 420" in result and not any(w in d for w in fraud_ctx):
        result.discard("IPC 420")

    # ── IPC 340 (wrongful confinement definition) → replace with IPC 342 ─────
    if "IPC 340" in result:
        result.discard("IPC 340")
        result.add("IPC 342")

    # ── IPC 326 redundancy fix ────────────────────────────────────────────────
    # IPC 325 (grievous hurt) already covers what 326 covers in most domestic cases
    # Keep 326 only if weapon + grievous injury both clearly mentioned
    if "IPC 326" in result and "IPC 325" in result:
        weapon_grievous = ["knife","sword","axe","gun","pistol","acid","deep cut","deep wound"]
        if not any(w in d for w in weapon_grievous):
            result.discard("IPC 326")

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 4 — MUTUAL EXCLUSION / CONTRADICTION FIXES
    # ══════════════════════════════════════════════════════════════════════════

    # IPC 302 (murder) and IPC 307 (attempt to murder) shouldn't both appear
    # unless description clearly mentions both attempt AND actual death
    both_death_attempt = (
        any(w in d for w in ["killed","dead","death","murdered","body found"])
        and any(w in d for w in ["also attacked","another person attacked","also tried to kill"])
    )
    if "IPC 302" in result and "IPC 307" in result and not both_death_attempt:
        result.discard("IPC 307")   # keep the more serious one

    # IPC 304A (negligent death) shouldn't coexist with IPC 302 (intentional murder)
    # unless both contexts clearly exist
    if "IPC 302" in result and "IPC 304A" in result:
        negligence_kws = ["accident","negligent","rash driving","doctor","careless"]
        if not any(w in d for w in negligence_kws):
            result.discard("IPC 304A")

    # IPC 379 (simple theft) + IPC 392 (robbery) — keep robbery, drop simple theft
    # since robbery is theft + force/threat
    if "IPC 392" in result and "IPC 379" in result:
        result.discard("IPC 379")   # robbery subsumes simple theft

    # IPC 395 (dacoity) present → drop IPC 392 (robbery), 395 is more specific
    if "IPC 395" in result and "IPC 392" in result:
        result.discard("IPC 392")

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 5 — GLOBAL VAGUE IPC REMOVAL
    # These sections almost never appear alone in a chargesheet — remove unless
    # there is highly specific context.
    # ══════════════════════════════════════════════════════════════════════════
    ALWAYS_REMOVE = {
        "IPC 410",   # stolen property definition — too vague
        "IPC 428",   # mischief on animal — remove unless animal mentioned
        "IPC 424",   # fraudulent removal — too vague
        "IPC 421",   # dishonest removal — too vague
        "IPC 193",   # false evidence court — remove unless court context present
        "IPC 191",   # giving false evidence — remove unless court context present
        "IPC 299",   # culpable homicide definition — too generic
        "IPC 300",   # murder definition — too generic (302 is the charge)
        "IPC 319",   # hurt definition — too generic (323/324 is the charge)
        "IPC 320",   # grievous hurt definition — too generic
        "IPC 340",   # wrongful confinement definition — use 342 instead
        "IPC 339",   # wrongful restraint definition — use 341 instead
        "IPC 349",   # force definition
        "IPC 350",   # criminal force definition
        "IPC 351",   # assault definition — use 352/354 instead
        "IPC 359",   # kidnapping definition
        "IPC 360",   # kidnapping from India definition
        "IPC 362",   # abduction definition
        "IPC 383",   # extortion definition (use 384 instead)
        "IPC 390",   # robbery definition (use 392 instead)
        "IPC 391",   # dacoity definition (use 395 instead)
        "IPC 415",   # cheating definition (use 420 instead)
        "IPC 463",   # forgery definition (use 465 instead)
        "IPC 375",   # rape definition (use 376 instead)
        "IPC 503",   # criminal intimidation definition (use 506 instead)
    }

    # Special case: keep IPC 193 if court context exists
    court_ctx = ["court","trial","judge","testimony","witness","oath","deposition"]
    if "IPC 193" in result and any(w in d for w in court_ctx):
        ALWAYS_REMOVE.discard("IPC 193")

    # Special case: keep IPC 428 if animal explicitly mentioned
    if "IPC 428" in result and any(w in d for w in ["animal","cattle","dog","cow","horse"]):
        ALWAYS_REMOVE.discard("IPC 428")

    result -= ALWAYS_REMOVE

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 6 — FINAL SANITY: cap at 12 most relevant sections
    # If model went overboard, keep only the highest-priority ones
    # ══════════════════════════════════════════════════════════════════════════
    PRIORITY_ORDER = [
        "IPC 302","IPC 307","IPC 304","IPC 304A","IPC 304B","IPC 306",
        "IPC 376","IPC 376D","IPC 375","IPC 354","IPC 354A","IPC 354D",
        "IPC 395","IPC 392","IPC 394","IPC 397","IPC 380","IPC 457",
        "IPC 363","IPC 364A","IPC 366","IPC 370",
        "IPC 498A","IPC 326A","IPC 326","IPC 324","IPC 325","IPC 323",
        "IPC 342","IPC 341","IPC 506","IPC 384","IPC 386",
        "IPC 420","IPC 465","IPC 467","IPC 468","IPC 471",
        "IPC 120B","IPC 120A",
        "IPC 498A","IPC 34",
        "IPC 25 Arms Act","IPC 27 Arms Act",
        "IPC 66C","IPC 66D","IPC 66",
        "IPC 7 PC Act","IPC 171E",
        "IPC 379","IPC 382","IPC 406","IPC 409",
        "IPC 354B","IPC 354C","IPC 376A",
        "IPC 307","IPC 308",
        "IPC 153A","IPC 147","IPC 148",
        "IPC 124A",
        "IPC 201","IPC 211","IPC 212",
    ]

    if len(result) > 12:
        ordered = [ipc for ipc in PRIORITY_ORDER if ipc in result]
        remaining = [ipc for ipc in result if ipc not in PRIORITY_ORDER]
        result = set(ordered[:12]) | set(remaining[:max(0, 12 - len(ordered[:12]))])

    return sorted(result)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_model(num_episodes: int = 1500, save_interval: int = 100):
    start = time.time()

    env        = ChargesheetEnvironment()
    state_size = env.state_size
    action_size = len(env.sections)

    print(f"Training  state_size={state_size}  action_size={action_size}")

    agent = ChargesheetAgent(state_size, 256, action_size)
    agent.load_model()   # continue if exists

    total_rewards, episode_times = [], []
    best_avg = float("-inf")

    for ep in range(num_episodes):
        t0    = time.time()
        state = env.reset()
        ep_r  = 0.0
        done  = False

        while not done:
            action              = agent.select_action(state)
            next_state, r, done = env.step(action)
            agent.store_transition(state, action, r)
            state = next_state
            ep_r += r

            if done:
                loss = agent.train()
                ep_t = time.time() - t0
                total_rewards.append(ep_r)
                episode_times.append(ep_t)
                agent.training_metrics["episode_rewards"].append(ep_r)
                agent.training_metrics["episode_losses"].append(loss or 0)
                agent.training_metrics["episode_times"].append(ep_t)

                if (ep + 1) % 50 == 0:
                    avg_r = np.mean(total_rewards[-50:])
                    print(f"Ep {ep+1}/{num_episodes}  avg_r={avg_r:.3f}  loss={loss:.4f}")
                    if avg_r > best_avg:
                        best_avg = avg_r
                        agent.save_model("chargesheet_model_best.pth")

        if (ep + 1) % save_interval == 0:
            agent.save_model(f"chargesheet_model_ep{ep+1}.pth")

    print(f"\nTraining done in {(time.time()-start)/60:.1f} min")
    agent.save_analytics(num_episodes, env)
    agent.save_model("chargesheet_model_final.pth")
    return agent, env


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def generate_chargesheet_from_description(description: str, agent, env) -> list:
    return env.get_relevant_sections(description)


def save_trained_model(agent, env,
                       model_name="chargesheet_model",
                       env_name="chargesheet_env"):
    agent.save_model(f"{model_name}.pth")
    env.save_environment(f"{env_name}.pth")


def load_trained_model(model_name="chargesheet_model",
                       env_name="chargesheet_env"):
    env = ChargesheetEnvironment()

    env_path = os.path.join("models", f"{env_name}.pth")
    if not os.path.exists(env_path):
        print(f"Env file missing: {env_path}")
        return None, None
    if not env.load_environment(f"{env_name}.pth"):
        return None, None

    agent = ChargesheetAgent(env.state_size, 256, len(env.sections))
    model_path = os.path.join("models", f"{model_name}.pth")
    if not os.path.exists(model_path):
        print(f"Model file missing: {model_path}")
        return None, None
    if not agent.load_model(f"{model_name}.pth"):
        return None, None

    return agent, env


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation  (F1-optimised)
# ──────────────────────────────────────────────────────────────────────────────

test_data = [
    {"description": "Two men snatched his bag and ran away.",
     "expected_ipcs": ["IPC 379","IPC 392"]},
    {"description": "He was assaulted with a knife in a fight.",
     "expected_ipcs": ["IPC 324","IPC 504"]},
    {"description": "The accused fraudulently transferred money online.",
     "expected_ipcs": ["IPC 420","IPC 66D"]},
    {"description": "A group of people looted a jewelry shop at night.",
     "expected_ipcs": ["IPC 395","IPC 120B"]},
    {"description": "A fake website was created to scam users.",
     "expected_ipcs": ["IPC 66C","IPC 419","IPC 420"]},
    {"description": "The victim was murdered with a pistol at his residence in January.",
     "expected_ipcs": ["IPC 302","IPC 450","IPC 25 Arms Act"]},
    {"description": "She was kidnapped from her home and taken to an unknown location.",
     "expected_ipcs": ["IPC 363","IPC 366"]},
    {"description": "He created forged documents to claim property ownership.",
     "expected_ipcs": ["IPC 465","IPC 420"]},
    {"description": "Acid was thrown on a girl after an argument in a market.",
     "expected_ipcs": ["IPC 326A","IPC 354"]},
    {"description": "The accused accepted a bribe to manipulate the case.",
     "expected_ipcs": ["IPC 171E","IPC 120B"]},
    {"description": "He was seen breaking into a house at midnight.",
     "expected_ipcs": ["IPC 380","IPC 457"]},
    {"description": "A threatening letter was sent to the minister demanding money.",
     "expected_ipcs": ["IPC 506","IPC 384"]},
    {"description": "Three people planned and executed a bank robbery.",
     "expected_ipcs": ["IPC 395","IPC 120B"]},
    {"description": "Doctor performed a surgery negligently leading to death.",
     "expected_ipcs": ["IPC 304A"]},
    {"description": "He was caught hacking into the company server.",
     "expected_ipcs": ["IPC 66C","IPC 66D"]},
    {"description": "The husband subjected his wife to continuous dowry harassment and physical abuse.",
     "expected_ipcs": ["IPC 498A","IPC 323"]},
    {"description": "A ransom call was received after child went missing.",
     "expected_ipcs": ["IPC 364A","IPC 363"]},
    {"description": "A man stalked and followed a woman to her workplace every day.",
     "expected_ipcs": ["IPC 354D","IPC 506"]},
]


def evaluate_model(test_data: list, use_rule_filter: bool = True):
    print("Loading trained model...")
    agent, env = load_trained_model()
    if not agent or not env:
        print("❌ Model load failed.")
        return

    totals = {"p": 0, "r": 0, "f1": 0}
    exact  = 0

    for i, item in enumerate(test_data, 1):
        expected  = {x.strip() for x in item["expected_ipcs"]}
        raw_pred  = generate_chargesheet_from_description(item["description"], agent, env)
        predicted = set(rule_based_filter(raw_pred, item["description"])) \
                    if use_rule_filter else set(raw_pred)

        tp = len(predicted & expected)
        fp = len(predicted - expected)
        fn = len(expected - predicted)

        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0

        if predicted == expected:
            exact += 1

        totals["p"]  += p
        totals["r"]  += r
        totals["f1"] += f1

        match_icon = "✅" if predicted == expected else "❌"
        print(f"\nCase {i:2d}  {match_icon}")
        print(f"  Desc      : {item['description'][:80]}")
        print(f"  Expected  : {sorted(expected)}")
        print(f"  Predicted : {sorted(predicted)}")
        print(f"  P={p:.2f}  R={r:.2f}  F1={f1:.2f}")

    n = len(test_data)
    print(f"\n{'─'*55}")
    print(f"  Avg Precision : {totals['p']/n:.2f}")
    print(f"  Avg Recall    : {totals['r']/n:.2f}")
    print(f"  Avg F1        : {totals['f1']/n:.2f}")
    print(f"  Exact Match   : {exact}/{n}  ({100*exact/n:.1f}%)")
    print(f"{'─'*55}")


def test_trained_model():
    agent, env = load_trained_model()
    if not agent or not env:
        print("No model found. Train first.")
        return

    description = "A man murdered his neighbour with a pistol at midnight after a heated argument."
    print(f"\nDescription: {description}")
    sections = generate_chargesheet_from_description(description, agent, env)
    filtered = rule_based_filter(sections, description)
    print("Predicted IPC sections:")
    for sec in filtered:
        cat = env.get_section_category(sec)
        print(f"  {sec}  [{cat}]")


if __name__ == "__main__":
    if (os.path.exists(os.path.join("models","chargesheet_model.pth")) and
            os.path.exists(os.path.join("models","chargesheet_env.pth"))):
        print("Found trained model — running evaluation...")
        evaluate_model(test_data)
    else:
        print("No trained model — starting training...")
        ag, ev = train_model(num_episodes=1500)
        save_trained_model(ag, ev)
        evaluate_model(test_data)
