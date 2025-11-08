import json
import os

# 输入和输出文件夹
input_dir = "/srv/nlprx-lab/share6/jhe478/tinker-cookbook/data"
output_dir = input_dir  # 可改为其他目录

# 遍历文件夹下所有 JSON 文件
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".json", ".jsonl"))

        with open(input_path, "r") as f:
            data = json.load(f)

        with open(output_path, "w") as f_out:
            for item in data:
                user_content = item["instruction"]
                if item.get("input"):
                    user_content = f"{item['instruction']}\n\n{item['input']}"

                conversation = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": item["output"]},
                    ]
                }

                f_out.write(json.dumps(conversation, ensure_ascii=False) + "\n")

        print(f"Converted: {filename} -> {os.path.basename(output_path)}")
