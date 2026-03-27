import requests
import sys
import uuid


text = " ".join(sys.argv[1:]) or input("Enter text: ")
r = requests.post("http://localhost:8000/speak", json={"text": text})
r.raise_for_status()
fname = f"output_{uuid.uuid4().hex[:8]}.mp3"
open(fname, "wb").write(r.content)
print(f"Saved: {fname}")
