import torch
from PIL import Image
from transformers.generation.streamers import TextIteratorStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def main(args):
    model_path = args.model_path
    device = args.device
    # 加载model，tokenizer
    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        **kwargs
        )
    # 初始化tokenizer
    model.initialize_tokenizer(tokenizer)

    prompts = [
        "这张图可能是在哪拍的？当去这里游玩时需要注意什么？",
        "Where might this picture have been taken? What should you pay attention to when visiting here?"
    ]

    import time
    time1 = time.time()
    
    for prompt in prompts:
        image = Image.open('./chang_chen.jpg').convert("RGB")
        # 对prompt进行分词，图像预处理
        input_ids, image_tensor, stopping_criteria = model.prepare_for_inference(
            prompt, 
            tokenizer, 
            image,
            return_tensors='pt')
        # 推理
        with torch.inference_mode():
            generate_ids = model.generate(
                inputs=input_ids.to(device),
                images=image_tensor.to(device),
                # do_sample=True,
                # temperature=0.2,
                # top_p=1.0,
                max_new_tokens=1024,
                # num_beams = 2,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                )
            # 截断generate_ids中的input_ids，然后解码为文本
            input_token_len = input_ids.shape[1]
            output = tokenizer.batch_decode(
                generate_ids[:, input_token_len:], 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
                )[0]
            print(output)
    
    time2 = time.time()
    print("cost seconds: ", time2 - time1)
    print("cost seconds per sample: ", (time2 - time1) / len(prompts))
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='DataCanvas/MMAlaya')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)


"""
CUDA_VISIBLE_DEVICES=0 python inference.py
"""