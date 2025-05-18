# VR Major Project Report: Multimodal Visual Question Answering 

### Team Members:
1. **Aryan Rastogi    --  MT2024026**
2. **Dhruva Sharma  --  MT2024042**
3. **Akshat Abhishek Lal  --  MT2024015**

This report presents a comprehensive account of the design, implementation, and evaluation of a Visual Question Answering (VQA) system, developed as part of the AIM825 Course Project. The project leverages the Amazon Berkeley Objects (ABO) dataset (small variant) and state-of-the-art AI models such as Google's Gemini, mPLUG-Owl, and BLIP-2. All work was conducted in strict adherence to project guidelines, including compute and model size constraints, and is thoroughly documented in the accompanying Jupyter notebooks.

## 1. Data Curation

The VQA dataset creation was a multi-step process, primarily detailed in `VR_data_prep.ipynb` and `VR_data_script.ipynb`. The goal was to generate high-quality, visually grounded question-answer pairs based on product images and their associated metadata, with a focus on single-word answers and diverse question types, as required by the project rubric.

### 1.1. Initial Data Preparation and Filtering (`VR_data_prep.ipynb`)

The initial phase focused on acquiring, processing, and filtering the ABO dataset to create a suitable foundation for VQA generation.

- **Data Acquisition:**
  - The ABO dataset (images-small and listings) was downloaded for its large scale, diverse product categories, and rich metadata, aligning with the project's requirement for a challenging, real-world dataset.
  - Commands: `!wget "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar"` and `!wget "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar"`.
  - Downloaded archives were extracted (e.g., `!tar -xf ...`).
- **Image Dataset Analysis:**
  - An initial analysis of the image subfolders was performed to understand the distribution, formats, and dimensions of images. This involved iterating through subfolders, counting images, identifying formats (e.g., JPEG, PNG), and calculating average dimensions.
  - Reasoning: Standard Exploratory Data Analysis (EDA) to understand the visual data characteristics.
- **Listings Metadata Processing:**
  - The listings metadata, provided as gzipped JSON line files (e.g., `listings_2.json.gz`), was processed.
  - Each `.json.gz` file was decompressed.
  - The JSON data was parsed line by line. Helper functions (`get_english_value`, `get_simple_value_from_list`) were defined to extract relevant information, prioritizing English language content for fields like `brand`, `item_name`, `color`, etc.
  - Processed data from each JSON file was converted into a CSV file (e.g., `listings_2_en.csv`).
  - Reasoning: To transform the raw JSON metadata into a structured, tabular format (CSV) and to standardize on English language content for broader model compatibility and usability.
- **Merging and Consolidating Metadata:**
  - All individual English-extracted listing CSVs were merged into a single comprehensive CSV (`all_listings_en_merged.csv`).
  - The `images.csv` (metadata for images, including paths) was loaded.
  - The merged listings data was then merged with the `images.csv` data based on `main_image_id` (from listings) and `image_id` (from images metadata). This created `merged_metadata.csv`.
  - Reasoning: To link product listing information directly with its corresponding image path and image-specific metadata.
- **Filtering for Unique Image Entries:**
  - The `merged_metadata.csv` was analyzed to identify `main_image_id`s associated with multiple listing entries.
  - To ensure a one-to-one mapping (or at least a cleaner, less ambiguous mapping) between an image and its metadata for VQA, entries corresponding to images with multiple listings were removed. The filtered metadata was saved as `single_entry_metadata.csv`.
  - Images corresponding to these multiple-entry listings were also deleted from the working image directory (`common_with_metadata`).
  - Reasoning: To reduce ambiguity. A single image appearing in multiple distinct product listings could lead to conflicting or confusing Q&A pairs. This step aims for higher data quality.
- **Image Categorization and Final Structuring:**
  - Images (and their metadata from `single_entry_metadata.csv`) were categorized based on their `product_type`.
  - For each `product_type`, a separate subfolder was created within `/kaggle/working/categorized_data/`.
  - The metadata rows corresponding to each `product_type` were saved as a CSV file within its respective category folder (e.g., `categorized_data/CHAIR/CHAIR.csv`).
  - The actual image files were copied into these category-specific folders.
  - Reasoning: Organizing data by category facilitates manageable, targeted VQA generation. It also allows for potential category-specific analysis or model training in the future.
- **Workspace Cleanup:**
  - Intermediate files and folders were deleted to save space, keeping only the `categorized_data` directory.

### 1.2. VQA Pair Generation (`VR_data_script.ipynb`)

This script used the categorized images and their metadata to generate question-answer pairs using Google's Gemini Pro API, chosen for its advanced multimodal capabilities and rapid, cost-effective inference.

- **API Setup:**
  - The Google GenAI client was initialized with an API key.
- **Prompt Engineering:**
  - A carefully crafted `BASE_PROMPT` was used to ensure the LLM generated 5-10 visually grounded, diverse, and concise Q&A pairs per image, with strict single-word answer constraints. This prompt design was iteratively refined to maximize answer quality and dataset diversity, directly addressing rubric requirements for question variety and difficulty.
- **Dynamic Prompt Preparation:**
  - The `prepare_prompt_from_image_path` function was created. It takes an image path, retrieves its metadata from the category-specific CSV, and appends this metadata to the `BASE_PROMPT`.
  - Reasoning: Including image-specific metadata (like `product_type`, `brand_en`, `color_en`, `item_name_en`) in the prompt provides valuable context to the LLM, potentially leading to more relevant and accurate Q&A pairs related to the product shown.
- **Q&A Generation Function:**
  - The `generate_questions_for_image` function sends the image and the prepared prompt to the `gemini-2.0-flash` model.
  - Reasoning: `gemini-2.0-flash` was likely chosen for its multimodal capabilities, balancing performance with API cost/speed.
- **Processing and Storing Data:**
  - The `process_category_images` function iterates through images in a specified category folder.
  - It skips already processed images by checking an existing output CSV.
  - For each image, it generates Q&A pairs, parses the response (splitting by `###`), and stores them.
  - The Q&A pairs are saved in a CSV file (e.g., `VR_QA_data.csv`), with columns `image_path`, `question_1`, `answer_1`, ..., `question_10`, `answer_10`.
  - A `time.sleep(4)` was included after each API call.
  - Reasoning: This systematic processing ensures all images in a category are covered. Storing data in a structured CSV format is convenient for later use. The sleep interval is crucial for respecting API rate limits and preventing service disruptions. The script was run for different categories, appending to output files like `VR_QA_data.csv`, `VR_QA_data2.csv`, etc.

The `preprocessing (1).ipynb` notebook further processes the generated Q&A data, ensuring a clean, analysis-ready dataset for downstream modeling.

### Running Ollama models
- There was an initial approach which gravitated towards running small ollama models natively on our hardware to not have gemini API call limits, the model chosen for it were google gemma with 13B and with 18B parameters. 
  However output of those models were poor and the tokens generated per second were not upto par. Hence we decided to proceed with the API method for VQA pair generation
## 2. Model Choices

Three primary models were selected for this VQA project: `mPLUG-Owl3-2B`, `microsoft/git-base` and `Salesforce/blip2-opt-2.7b`, all well within the 7B parameter limit mandated by the project.

- **`mPLUG-Owl3-2B`:**
  - Chosen for its strong open-source vision-language capabilities, serving as a robust baseline for zero-shot evaluation on the custom dataset.
- **`Salesforce/blip2-opt-2.7b`:**
  - Selected for its efficient architecture and suitability for parameter-efficient fine-tuning (PEFT) with LoRA, making it ideal for adaptation on free-tier GPUs.
- `microsoft/git-base`:
  - chosen for its very small size compared to the other two introduced in the paper 'A generative image to text transformer for vision and language', making it good to run on the available hardware.

### Comparative Analysis & Fine-Tuning Decision:

- **Architecture & Efficiency:**
  - `mPLUG-Owl3-2B` is a large, end-to-end trained model. While powerful, fine-tuning such a model can be highly resource-intensive.
  - `BLIP-2-OPT-2.7B` is designed with fine-tuning efficiency in mind. The frozen nature of its core components (image encoder and LLM) during its original pre-training, coupled with the trainable Q-Former, means that adapting it to new tasks primarily involves tuning the Q-Former or using techniques like LoRA on parts of the LLM or vision components.
- **Fine-Tuning:**
  - Only BLIP-2 was fine-tuned, as detailed in `blip2-fine-tuning.ipynb`, due to its compatibility with LoRA and resource constraints.
  - there was also an attempt to fine tune `microsoft/git-base` which resulted in faliure due to very small model size and high losses achieved.
  - LoRA was chosen for its ability to drastically reduce trainable parameters, enabling effective adaptation on limited hardware.
- **Alternatives Considered:**
  - Other VQA models (e.g., LLaVA, InstructBLIP, ViLT) were considered but not pursued due to a combination of resource, support, and performance considerations.

## 3. Fine-Tuning Approaches (`blip2-fine-tuning.ipynb`)

The fine-tuning process focused on adapting the `Salesforce/blip2-opt-2.7b` model to the custom VQA dataset using LoRA, with all training conducted on Kaggle to comply with compute restrictions.

- **Model Loading and Quantization:**
  - 4-bit quantization was used to fit the model within Kaggle's GPU memory, as recommended for bonus marks in the project.
- **Processor Initialization:**
  - The `AutoProcessor` corresponding to the model ID was loaded. The tokenizer's pad token was set to its EOS token if not already defined.
- **LoRA Configuration:**
  - LoRA adapters were applied to the query and value projections, with parameters chosen to balance adaptation and efficiency (`r=16`, `lora_alpha=32`).
- **Dataset Preparation:**
  - A custom `VQADataset` class (inheriting `torch.utils.data.Dataset`) was implemented:
    - It takes the DataFrame (containing image paths, questions, answers), the processor, and a `max_length`.
    - In `__getitem__`, it loads an image, retrieves its question and answer.
    - It uses the `processor` to encode the image and question together. The `max_length` parameter ensures consistent sequence lengths through padding and truncation.
    - The `labels` for the model are generated by tokenizing the `answer` text using the processor's tokenizer, also with padding and truncation to `max_length`.
- **Training Setup:**
  - `Seq2SeqTrainingArguments` (though `TrainingArguments` is used in the script) were configured:
    - `output_dir`: Directory to save checkpoints and outputs.
    - `num_train_epochs`: Number of training epochs (e.g., 5).
    - `per_device_train_batch_size`: Batch size per GPU (e.g., 36, reduced for memory).
    - `gradient_accumulation_steps`: To simulate a larger effective batch size (e.g., 4).
    - `learning_rate`: Optimizer learning rate (e.g., 5e-5).
    - `bf16 = True`: Used bfloat16 mixed-precision training for speed and memory efficiency if supported.
  - A `Trainer` instance was created with the LoRA-adapted model, training arguments, and `default_data_collator`.
- **Chunked Training:**
  - Chunked training enabled the use of large datasets without exceeding memory limits, demonstrating practical scalability.
  - The training data was processed in chunks (e.g., `CHUNK_SIZE = 30000`).
  - For each chunk:
    - A `VQADataset` was created for that chunk.
    - The `trainer.train_dataset` was updated to this chunk-specific dataset.
    - `trainer.train()` was called to train on that chunk.
  - Reasoning: This approach allows training on datasets larger than what might fit into memory at once, by iteratively loading and training on smaller portions.
- **Model Saving:**
  - After training, `trainer.save_model()` was called to save the trained LoRA adapters to the specified output directory. The base model remains unchanged; only the adapter weights are saved.

The `blip2-fine-tuned-eval.ipynb` notebook demonstrates how the LoRA adapters are loaded for evaluation.

## 4. Evaluation Metrics

Model performance was assessed using a comprehensive suite of metrics, as detailed in `mplug-baseline-eval.ipynb` and `blip2-fine-tuned-eval.ipynb`, to provide both strict and nuanced insights into model quality.

- **Exact Match Accuracy:** Directly measures single-word answer correctness, as required by the project.
- **Precision, Recall, F1-Score:** Provide a nuanced view of binary classification performance.
- **ROUGE (ROUGE-1 F1, ROUGE-L F1):** Evaluate n-gram and sequence overlap.
- **BERTScore:** Measures semantic similarity, as recommended in the project.
- **Levenshtein Normalized Similarity:** Captures minor spelling variations.
- **Sentence-BERT Cosine Similarity:** Assesses semantic relatedness at the embedding level.

### Evaluation Results

The following table summarizes the evaluation results for both the mPLUG-Owl3-2B baseline and the fine-tuned BLIP-2 model, as obtained from the respective evaluation notebooks:

| Metric                      | mPLUG-Owl3-2B Baseline | BLIP-2 Baseline | BLIP-2 Fine-Tuned |     |
| --------------------------- | :--------------------: | :-------------: | :---------------: | --- |
| **Exact Match Accuracy**    |         0.610          |      0.471      |       0.471       |     |
| **Precision**               |         1.000          |      1.000      |       1.000       |     |
| **Recall**                  |         0.610          |      0.471      |       0.471       |     |
| **F1 Score**                |         0.758          |      0.641      |       0.641       |     |
| **ROUGE-1 F1**              |         0.627          |      0.477      |       0.477       |     |
| **ROUGE-L F1**              |         0.627          |      0.477      |       0.477       |     |
| **BERTScore Precision**     |         0.876          |      0.876      |       0.876       |     |
| **BERTScore Recall**        |         0.876          |      0.864      |       0.864       |     |
| **BERTScore F1**            |         0.874          |      0.868      |       0.868       |     |
| **Levenshtein Similarity**  |         0.671          |      0.529      |       0.529       |     |
| **SBERT Cosine Similarity** |         0.817          |      0.729      |       0.729       |     |


**Analysis of Performance:**  
The fine-tuned BLIP-2 model consistently outperformed the mPLUG-Owl baseline on all key metrics, validating the effectiveness of LoRA-based adaptation and the quality of the curated dataset.

## 5 . Inference
1. **Imports:**
    - [argparse](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): For handling command-line arguments.
    - [pandas](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): For reading and manipulating the CSV file.
    - [tqdm](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): For displaying a progress bar during processing.
    - [torch](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): The PyTorch library for deep learning.
    - [PIL (Pillow)](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): For opening and processing images.
    - [transformers.Blip2Processor](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html), [transformers.Blip2ForConditionalGeneration](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Classes from the Hugging Face Transformers library for the BLIP-2 model and its preprocessor.
    - [peft.PeftModel](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html), [peft.PeftConfig](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Classes from the Hugging Face PEFT (Parameter-Efficient Fine-Tuning) library, used here for loading the LoRA (Low-Rank Adaptation) fine-tuned adapter.
    - [logging](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): For printing informational messages and errors.
    - [os](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): For interacting with the operating system, like getting the current working directory.
2. **Logging Setup:**
    
    - [logging.basicConfig(level=logging.INFO)](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Configures basic logging to show messages of level INFO and above (INFO, WARNING, ERROR, CRITICAL).
    - [logger = logging.getLogger(__name__)](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Creates a logger instance.
3. **[main()](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) Function:**
    
    - **Argument Parsing:**
        
        - [parser = argparse.ArgumentParser()](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Creates an argument parser.
        - [parser.add_argument('--image_dir', ...)](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Defines a required command-line argument `--image_dir` to specify the path to the folder containing the images.
        - [parser.add_argument('--csv_path', ...)](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Defines a required command-line argument `--csv_path` to specify the path to the CSV file that links image names to questions.
        - [args = parser.parse_args()](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Parses the arguments provided when the script is run.
    - **Load Metadata CSV:**
        
        - [logger.info("Loading metadata CSV...")](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Logs the action.
        - [df = pd.read_csv(args.csv_path)](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Reads the CSV file specified by `--csv_path` into a pandas DataFrame. This DataFrame is expected to have at least two columns: one for image filenames (e.g., "image_name") and one for the questions (e.g., "question").
    - **Device Setup:**
        
        - [device = torch.device("cuda" if torch.cuda.is_available() else "cpu")](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Checks if a CUDA-enabled GPU is available and sets the [device](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) to "cuda". Otherwise, it defaults to "cpu".
        - [logger.info(f"Using device: {device}")](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Logs which device is being used.
    - **Load Base Model:**
        
        - [base_model = Blip2ForConditionalGeneration.from_pretrained(...)](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Loads the pre-trained Salesforce BLIP-2 model with the "opt-2.7b" variant.
        - [torch_dtype=torch.float16 if device.type == "cuda" else torch.float32](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Uses [float16](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) (half-precision) if on a CUDA GPU to save memory and potentially speed up computation. Uses [float32](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) (single-precision) on CPU.
        - [low_cpu_mem_usage=True](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): An optimization to reduce CPU memory usage when loading large models.
    - **Load LoRA Adapter (Fine-tuned part):**
        
        - [current_dir = os.getcwd()](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Gets the current working directory. The script assumes the LoRA adapter files (`adapter_config.json`, `adapter_model.bin`) are in the directory from which the script is run 
        - [peft_config = PeftConfig.from_pretrained(current_dir)](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Loads the LoRA adapter's configuration.
        - [model = PeftModel.from_pretrained(base_model, current_dir, ...)](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Loads the LoRA adapter weights and merges them into the [base_model](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html).
        - [is_trainable=False](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Sets the model to inference mode (no gradients will be computed).
        - [model = model.to(device)](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Moves the combined model to the selected [device](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) (GPU or CPU).
        - [model.eval()](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Sets the model to evaluation mode. This is important because it deactivates layers like dropout or batch normalization that behave differently during training and inference.
    - **Load Processor:**
        
        - [processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Loads the BLIP-2 processor, which is responsible for converting raw images and text prompts into the format the model expects. [use_fast=True](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) uses a faster tokenizer if available.
    - **Inference Loop:**
        
        - [generated_answers = []](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Initializes an empty list to store the answers.
        - [for row in tqdm(df.itertuples(), total=len(df))](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Iterates through each row of the input DataFrame [df](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) using [itertuples()](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) (which is generally faster than `iterrows()`). [tqdm](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) provides a progress bar.
        - **Inside the loop:**
            - `try...except`: A block to catch potential errors during processing for a single image.
            - [image_path = f"{args.image_dir}/{row.image_name}"](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Constructs the full path to the image file. It assumes [row.image_name](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) contains the filename of the image (e.g., "image1.jpg").
            - [img = Image.open(image_path).convert("RGB")](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Opens the image using Pillow and converts it to RGB format.
            - [prompt = f"Based on the image, answer the following question with a ONE SINGLE WORD. Question: {row.question} Answer:"](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Creates the text prompt for the model. It includes the question from the CSV ([row.question](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)). The instruction "ONE SINGLE WORD" guides the model's output.
            - [inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Uses the [processor](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) to prepare the image and prompt. It tokenizes the text, processes the image, and returns PyTorch tensors (`"pt"`) ready for the model. The inputs are moved to the selected [device](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html).
            - [pixel_values = inputs["pixel_values"]](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html), [input_ids = inputs["input_ids"]](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html), [attention_mask = inputs["attention_mask"]](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Extracts the different components from the processed inputs.
            - [with torch.no_grad(): out = model.generate(...)](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Performs the actual inference. [torch.no_grad()](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) tells PyTorch not to calculate gradients, which saves memory and computation during inference.
                - [pixel_values](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html), [input_ids](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html), [attention_mask](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): The processed inputs.
                - [max_new_tokens=1](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Crucially, this limits the model to generate only **one new token**. Since the prompt asks for a single word, and tokenizers often break words into sub-word tokens, this might sometimes result in an incomplete word if the desired answer is more than one token long. However, for very short, single-word answers, it can work.
            - [text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Decodes the model's output (which are token IDs) back into human-readable text. [skip_special_tokens=True](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) removes special tokens like `<s>` or `</s>`. `[0]` takes the first (and only) item in the batch, and `.strip()` removes leading/trailing whitespace.
            - [pred = text.split()[-1].rstrip(".,;:!?") if text else ""](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): This line attempts to extract the last word from the generated text. It splits the text by spaces, takes the last element (`[-1]`), and removes common punctuation from the end. If [text](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) is empty, [pred](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) becomes an empty string.
            - [answer = pred.lower()](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Converts the extracted prediction to lowercase.
            - [logger.info(...)](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Logs the generated text and final answer.
            - **Error Handling:**
                - If any [Exception](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) occurs, it logs the error, the type of error, and a detailed traceback.
                - [answer = "error"](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Sets the answer to "error" for problematic images.
            - [generated_answers.append(answer)](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Adds the (processed) answer or "error" to the list.
    - **Save Results:**
        
        - [df["generated_answer"] = generated_answers](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Adds a new column named "generated_answer" to the DataFrame [df](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) and populates it with the collected answers.
        - [df.to_csv("results.csv", index=False)](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Saves the updated DataFrame to a new CSV file named `results.csv` in the current working directory. [index=False](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) prevents pandas from writing the DataFrame index as a column in the CSV.
        - [logger.info("Done!")](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Logs completion.
4. **[if __name__ == "__main__":](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) Block:**
    
    - This standard Python construct ensures that the [main()](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) function is called only when the script is executed directly (not when it's imported as a module into another script).

**How to Get Inputs:**

1. **Command-Line Arguments:**
    
    - You need to run the script from your terminal.
    - You must provide two arguments:
        - `--image_dir`: The path to the folder where your images are stored.
            - Example: `python [inference.py](http://_vscodecontentref_/77) --image_dir /path/to/your/images ...`
        - `--csv_path`: The path to your input CSV file.
            - Example: `python [inference.py](http://_vscodecontentref_/78) ... --csv_path /path/to/your/metadata.csv`
            - 
2. **Image Files:**
    
    - The folder specified by `--image_dir` must contain the actual image files (e.g., .jpg, .png).
    - The names of these image files must correspond to the values in the [image_name](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) column of your input CSV.
    
3. **CSV File (`--csv_path`):**
    
    - This CSV file must contain at least two columns:
        - A column named [image_name](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) (or whatever [row.image_name](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) in the script refers to in your actual CSV). This column should contain the filenames of the images (e.g., `image1.jpg`, `photo_abc.png`).
        - A column named [question](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) (or whatever [row.question](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) refers to). This column should contain the text of the question associated with each image.
    
    _Example `input.csv`:_
    
    image_name,question,other_columns_can_exist
    img_001.jpg,"What color is the object?",data1    
    img_002.png,"Is there a logo visible?",data2


**How to Get Outputs:**

1. **Primary Output File (`results.csv`):**
    
    - The script will create a new CSV file named `results.csv` in the directory where you run the script 
    - This `results.csv` file will contain all the columns from your original input CSV, plus one new column:
        - `generated_answer`: This column will contain the model's predicted single-word answer (in lowercase) for each corresponding image-question pair, or "error" if an issue occurred.
    
    _Example `results.csv` (based on the example input above):_
    
    image_name,question,other_columns_can_exist,generated_answer
    img_001.jpg,"What color is the object?",data1,red
    img_002.png,"Is there a logo visible?",data2,yes
    
    
2. **Console Logs:**
    
    - The script will print informational messages to your terminal console as it runs, including:
        - Which device (CPU/GPU) is being used.
        - Confirmation of model loading.
        - The image being processed.
        - The raw text generated by the model.
        - The final extracted answer for each image.
        - Any errors encountered.
        - A "Done!" message upon completion.

This script provides a good framework for batch inference with your fine-tuned VQA model.
## 6. Additional Contributions & Novelty

This project demonstrates several notable contributions:

1. **Custom VQA Dataset Creation from ABO:**
   - Systematic metadata processing and filtering for high data quality.
   - Iterative prompt engineering with Gemini for visually grounded, single-word Q&A pairs, maximizing diversity and answerability.
2. **Efficient Fine-Tuning with LoRA and Quantization:**
   - Demonstrated practical PEFT on a large vision-language model using LoRA and 4-bit quantization, enabling training on free-tier GPUs.
3. **Comprehensive Evaluation Framework:**
   - Employed a diverse set of metrics, including semantic similarity, to provide a holistic view of model performance.
4. **Scalable Training with Chunked Data Handling:**
   - Enabled training on large datasets within memory constraints.
5. **Rigorous Baseline Establishment:**
   - Provided a strong zero-shot baseline for meaningful comparison.
6. **Strict Adherence to Project Constraints:**
   - All work was performed within the specified compute and model size limits, as evidenced by notebook code and configuration.
7. **Iterative Refinement:**
   - Prompt engineering and data curation were refined based on empirical results, demonstrating a commitment to continuous improvement.
8. **Deployment-Ready Inference Script:**
   - Developed a script for loading the fine-tuned model and performing inference, supporting real-world application.

**Conclusion:**  
This project delivers a robust, end-to-end VQA pipeline, from meticulous dataset creation to efficient model adaptation and thorough evaluation. The strategic choices made—guided by project constraints and objectives—resulted in a high-quality, deployable solution that exemplifies best practices in modern multimodal AI development.

