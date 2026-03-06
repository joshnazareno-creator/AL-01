import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate("ServiceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

doc_ref = db.collection("al01_memory").document("python_boot")
doc_ref.set({
    "event": "python_connected",
    "status": "operational"
})

print("AL-01 is now connected to Firestore.")