.PHONY: install train eval demo test serve docker
install:  ; pip install -r requirements.txt
train:    ; python -m floodwm.train
eval:     ; python -m floodwm.evaluate
demo:     ; python scripts/run_demo.py
test:     ; pytest -q
serve:    ; uvicorn floodwm.api:app --port 8000 --reload
docker:   ; docker build -t floodwm . && docker run -p 8000:8000 floodwm
