#!/bin/bash
# gemma3-4b-it
uv run main.py --video_dir falling_off_chair --model_id google/gemma-3-4b-it
uv run llm_judge.py --video_dir falling_off_chair --model_id google/gemma-3-4b-it

uv run main.py --video_dir falling_off_bike --model_id google/gemma-3-4b-it
uv run llm_judge.py --video_dir falling_off_bike --model_id google/gemma-3-4b-it

uv run main.py --video_dir face_planting --model_id google/gemma-3-4b-it
uv run llm_judge.py --video_dir face_planting --model_id google/gemma-3-4b-it


# gemma3-12b-it
uv run main.py --video_dir falling_off_chair --model_id google/gemma-3-12b-it
uv run llm_judge.py --video_dir falling_off_chair --model_id google/gemma-3-12b-it

uv run main.py --video_dir falling_off_bike --model_id google/gemma-3-12b-it
uv run llm_judge.py --video_dir falling_off_bike --model_id google/gemma-3-12b-it

uv run main.py --video_dir face_planting --model_id google/gemma-3-12b-it
uv run llm_judge.py --video_dir face_planting --model_id google/gemma-3-12b-it
