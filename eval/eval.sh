python eval.py \
  --model "Qwen/Qwen3-8B-Base" \
  --dataset "math500" \
  --query_language "English" \
  --reasoning_language "English" \
  --output_path "./results"

python eval.py \
  --model "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset "math500" \
  --query_language "English" \
  --reasoning_language "English" \
  --output_path "./results"
  
python eval_tinker.py \
  --model "tinker://53ae2176-89aa-4f21-a2d9-af940c6cbf2e/sampler_weights/final" \
  --base_model_for_tokenizer "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset "math500" \
  --query_language "English" \
  --reasoning_language "English" \
  --output_path "./results"
  
# --sample \