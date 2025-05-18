import argparse
import pandas as pd
from tqdm import tqdm
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel, PeftConfig
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    # Load metadata CSV
    logger.info("Loading metadata CSV...")
    df = pd.read_csv(args.csv_path)

    # Load base model and processor
    logger.info("Setting up device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load your fine-tuned model
    logger.info("Loading fine-tuned model...")
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Load the adapter config and model
    logger.info("Loading adapter config and model...")
    current_dir = os.getcwd()
    peft_config = PeftConfig.from_pretrained(current_dir)
    model = PeftModel.from_pretrained(
        base_model,
        current_dir,
        config=peft_config,
        is_trainable=False
    )
    model = model.to(device)
    model.eval()

    # Load processor
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)
    logger.info("Model loaded successfully!")

    generated_answers = []
    for row in tqdm(df.itertuples(), total=len(df)):
        try:
            image_path = f"{args.image_dir}/{row.image_name}"
            logger.info(f"Processing image: {image_path}")
            
            # Load and process image
            img = Image.open(image_path).convert("RGB")
            
            # Prepare prompt and inputs
            prompt = f"Based on the image, answer the following question with a ONE SINGLE WORD. Question: {row.question} Answer:"
            inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
            
            # Extract input components
            pixel_values = inputs["pixel_values"]
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # Generate prediction
            with torch.no_grad():
                out = model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1
                )
            
            # Decode and process the output
            text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            logger.info(f"Generated text: {text}")
            
            # Extract the last word as the prediction
            pred = text.split()[-1].rstrip(".,;:!?") if text else ""
            answer = pred.lower()
            logger.info(f"Final answer: {answer}")
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            answer = "error"
            
        generated_answers.append(answer)

    # Save results
    logger.info("Saving results...")
    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)
    logger.info("Done!")

if __name__ == "__main__":
    main()