#!/bin/bash
cd backend/final\ stuff
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
