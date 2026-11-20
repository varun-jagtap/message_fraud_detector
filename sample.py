"""Deterministic sample data generator for smoke tests and demos."""

import csv
import json
import random
from pathlib import Path
from typing import Iterable, Iterator, List


FRAUD_TEMPLATES = (
    "URGENT! Your bank account has been suspended. Verify your password and OTP immediately: https://secure-bank.example/reset",
    "Congratulations, you won a reward. Pay a Rs.250 processing fee by gift card and send the code now.",
    "Police case detected. Transfer money by crypto today or legal action will follow.",
    "Amazon security alert: login and PIN required immediately. Call 1-800-555-0199.",
    "Your income tax refund is waiting. Send your PAN, SSN and bank login within 10 minutes.",
    "Limited offer: claim prize before it expires, send Bitcoin payment to verify.",
    "Your package will be deleted unless you verify account credentials at http://pay.example.",
    "Final warning! Your account is locked. Text password and OTP to +91 90000 11122.",
    "Your UPI ID will be blocked today. Approve this collect request and share the OTP with our agent.",
    "KYC expired. Send Aadhaar, PAN and debit card details to reactivate your Bank of Maharashtra account.",
    "You have been selected for a Rs.50,000 cash prize. Pay GST charges through this link to receive it.",
    "Electricity disconnection notice: pay Rs.1,999 immediately at http://bill-pay.example or your meter will be sealed.",
    "Your SIM will be deactivated in 30 minutes. Confirm your OTP and remote-access code now.",
    "Courier delivery failed. Pay Rs.35 redelivery charges and enter your card PIN at http://delivery.example.",
    "Dear customer, suspicious activity detected. Move funds to this safe account and send the transaction screenshot.",
    "Your FASTag wallet is suspended. Complete verification with card number, CVV and OTP at http://fastag-help.example.",
    "You have won a festival bumper draw. Share your bank login and processing fee to release the prize.",
    "Income tax department notice: arrest warrant issued. Transfer money immediately to avoid legal action.",
    "Your mobile number won a lucky draw. Call this number and provide the OTP to claim your reward.",
    "Account closure pending. Send a signed cheque, PIN and net-banking password to prevent permanent closure.",
)

LEGITIMATE_TEMPLATES = (
    "Booking ID: WJPW7QP. Seats: ROYAL GO - B11 for Rajini: The Jailer 2 (Hindi) on Mon, 19 Oct, 2026 12:15pm at City Pride, Nyati Plaza: Kharadi (SCREEN7). Ticket(s): 1 Ticket Price: Rs.250.00 Download M-ticket from https://bmsurl.co/BMSHOW/djzaDjm1jj. - BookMyShow",
    "Your appointment is confirmed for Tuesday at 2 PM at City Hospital.",
    "The team meeting agenda is attached for your review. Meeting starts at 10 AM.",
    "Thanks for your purchase. This message confirms our scheduled delivery for tomorrow.",
    "Please remember to submit the project status report by Friday.",
    "Dinner is at 7 PM at the restaurant near the station.",
    "The library loan is due next month. You can renew it online from the library portal.",
    "Your monthly account statement is ready in the customer portal.",
    "Thanks for contacting support. We will reply during normal business hours.",
    "Dear Customer, your Bank of Maharashtra account statement for September is available in net banking.",
    "Dear Customer You have not operated your account no ***1234 for more than 90 days. Please operate your account from being classified as inoperative. -Bank of Maharashtra",
    "Dear Customer, your cheque number 004218 has been presented for clearing. -Bank of Maharashtra",
    "Your UPI payment of Rs.450 to Metro Stores was successful. UTR: 628194730112.",
    "Your electricity bill payment of Rs.1,240 was received on 05 Oct 2026. Thank you.",
    "Your train ticket PNR 4819201556 is confirmed. Travel date: 22 Oct 2026.",
    "Your flight booking to Delhi is confirmed. Booking reference: AI7K2P. Check-in opens 48 hours before departure.",
    "Your order #OD45821 has been shipped and will arrive by Friday.",
    "Your salary credit of Rs.68,500 has been processed to account ending ****1234.",
    "Your insurance premium of Rs.3,250 is due on 28 Oct 2026. Pay through the official app.",
    "Your school fee receipt for student ID STU2048 has been generated successfully.",
    "Your parcel from CityBooks has been delivered. Thank you for shopping with us.",
)

SUSPICIOUS_TEMPLATES = (
    "Please verify your account details when you have time.",
    "A payment request is pending; check the bank portal before proceeding.",
    "Limited-time offer available; review the reward terms.",
    "Dear Customer, beware of scams. Never share OTP, PIN, CVV or passwords. Avoid unknown links and calls. In case of fraud report on 1930 or 18002334526. -Bank of Maharashtra",
    "Your account has been inactive for more than 90 days. Please operate the account to avoid inoperative status. -Bank of Maharashtra",
    "Your KYC details may need an update. Please visit the official branch or banking application.",
    "A new beneficiary was added to your account. Contact the bank through the number on your debit card if this was not you.",
    "Your card payment could not be completed. Review the transaction in the official banking application.",
    "A refund of Rs.799 is being processed. Check your merchant account for the final status.",
    "Your delivery address needs confirmation before dispatch. Use the official shopping application.",
    "A login was detected from a new device. Review your recent activity and change your password if needed.",
    "Your reward points are nearing expiry. Review the offer in the official customer portal.",
    "A subscription renewal is scheduled. Check the service application to manage your plan.",
    "Your mobile recharge is pending. Verify the status in the official operator application.",
    "A bank service update is available. Read the notice in your secure net-banking inbox.",
    "Your train booking has a travel advisory. Check the railway application for updated details.",
    "Your insurance policy requires a document review. Contact the insurer through its official website.",
    "A payment link was generated for your order. Confirm the merchant name before paying.",
    "Your account profile has incomplete information. Update it through the official branch or app.",
    "A courier attempt was recorded. Check the tracking page using the official courier website.",
)


def generate_records(count: int = 100, seed: int = 7) -> List[dict]:
    if count < 1:
        raise ValueError("count must be at least 1")
    if count > 100000:
        raise ValueError("count must not exceed 100000")
    rng = random.Random(seed)
    records = []
    pools = (
        ("fraud", FRAUD_TEMPLATES),
        ("legitimate", LEGITIMATE_TEMPLATES),
        ("suspicious", SUSPICIOUS_TEMPLATES),
    )
    for index in range(count):
        label, pool = rng.choice(pools)
        text = rng.choice(pool)
        mutation = index % 5
        if mutation == 1:
            text = "Reminder: " + text
        elif mutation == 2:
            text = text + " Please act soon."
        elif mutation == 3:
            text = text.replace("account", "profile")
        records.append({
            "message_id": "sample-{:05d}".format(index + 1),
            "text": text,
            "label": label,
            "source": "sample_generator",
        })
    return records


def write_jsonl(records: Iterable[dict], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def write_csv(records: Iterable[dict], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    fieldnames = ["message_id", "text", "label", "source"]
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})
