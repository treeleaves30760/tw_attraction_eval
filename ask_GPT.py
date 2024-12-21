import os
from dotenv import load_dotenv
import json
import base64
import argparse
import re
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime
from openai import OpenAI

load_dotenv()

class ImageAnalyzer:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def extract_location_name(self, image_path: str) -> str:
        """Extract location name from image path."""
        # Extract filename without extension
        filename = os.path.splitext(os.path.basename(image_path))[0]
        # Split by hyphen and take the first part
        location_name = filename.split('-')[0]
        return location_name

    def check_location_in_response(self, location_name: str, response: str) -> Tuple[bool, str]:
        """
        Check if the location name appears in the response.
        Returns a tuple of (is_found, details)
        """
        # Remove common punctuation from both strings for comparison
        def clean_text(text):
            # Remove punctuation and whitespace
            return re.sub(r'[^\w\s]', '', text).strip().lower()
        
        clean_location = clean_text(location_name)
        
        if clean_location in response:
            return True, "Location name found in response"
        else:
            return False, f"Location '{location_name}' not found in response"

    def analyze_image(self, image_path: str, model: str, question: str = "請問圖片中的景點是哪裡？") -> Dict:
        """Analyze a single image using specified model."""
        try:
            base64_image = self.encode_image(image_path)
            location_name = self.extract_location_name(image_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=300
            )
            
            response_text = response.choices[0].message.content
            location_found, analysis_details = self.check_location_in_response(location_name, response_text)
            
            return {
                "filename": os.path.basename(image_path),
                "expected_location": location_name,
                "response": response_text,
                "location_found": location_found,
                "analysis_details": analysis_details,
                "status": "success",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
                    
        except Exception as e:
            print(e)
            return {
                "filename": os.path.basename(image_path),
                "expected_location": self.extract_location_name(image_path),
                "response": f"Error: {str(e)}",
                "location_found": False,
                "analysis_details": "Error occurred during processing",
                "status": "error",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    def process_images(self, folder_path: str) -> Dict[str, List[Dict]]:
        """Process all images in the given folder using both models."""
        results = {"gpt-4o": [],}
        supported_formats = {'.jpg', '.jpeg', '.png', '.gif'}
        
        image_files = [
            f for f in Path(folder_path).iterdir()
            if f.is_file() and f.suffix.lower() in supported_formats
        ]
        
        # Process images for each model
        for model in ["gpt-4o"]:
            for image_file in image_files:
                result = self.analyze_image(str(image_file), model)
                results[model].append(result)
                print(f"Processed {os.path.basename(image_file)} with {model}")
                # Print location verification result
                print(f"Location verification: {result['analysis_details']}")
                print(f"Response: {result['response']}")
        
        return results

    def save_results(self, results: Dict[str, List[Dict]], output_folder: str, file_name: str):
        """Save results to separate JSON files for each model."""
        os.makedirs(output_folder, exist_ok=True)
        
        for model, model_results in results.items():
            # Calculate statistics
            total_images = len(model_results)
            successful_location_matches = sum(1 for r in model_results if r['location_found'])
            
            output_file = os.path.join(output_folder, f"{file_name}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        "model": model,
                        "total_images": total_images,
                        "successful_location_matches": successful_location_matches,
                        "location_match_rate": f"{(successful_location_matches/total_images)*100:.2f}%",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "results": model_results
                    },
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            print(f"Saved results for {model} to {output_file}")
            print(f"Location match rate: {(successful_location_matches/total_images)*100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Analyze images using GPT-4 series models')
    parser.add_argument('--input-folder', help='Path to the folder containing images', default='./images')
    parser.add_argument('--output-folder', help='Path to the output folder for JSON files', default='./results')
    parser.add_argument('--file-name', help='Name of the output JSON file', default=f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    parser.add_argument('--api-key', help='OpenAI API key', default=os.getenv('OPENAI_API_KEY'))
    
    args = parser.parse_args()
    
    analyzer = ImageAnalyzer(args.api_key)
    results = analyzer.process_images(args.input_folder)
    analyzer.save_results(results, args.output_folder, args.file_name)
    print("Analysis complete.")

if __name__ == "__main__":
    main()