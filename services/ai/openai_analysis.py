#!/usr/bin/env python3
"""
OpenAI Analysis Service

This module provides image analysis functionality using OpenAI's vision models.
It can analyze flood overhead images along with satellite imagery to provide
insights about flood damage, building conditions, and environmental impact.

Usage:
    from services.ai.openai_analysis import OpenAIAnalyzer
    
    analyzer = OpenAIAnalyzer()
    result = analyzer.analyze_flood_images(flood_image_path, satellite_image_path)
"""

import os
import base64
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import io

import openai
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIAnalyzer:
    """Handles OpenAI vision analysis of flood and satellite images."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI analyzer.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        
    def _convert_and_encode_image(self, image_path: str) -> str:
        """
        Convert image to PNG format (if needed) and encode as base64.
        
        OpenAI Vision API supports:
        - PNG (.png)
        - JPEG (.jpeg and .jpg) 
        - WEBP (.webp)
        - Non-animated GIF (.gif)
        - Size limit: Up to 50MB per image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Check file size (50MB limit)
            file_size_mb = image_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 50:
                logger.warning(f"Image file {image_path} is {file_size_mb:.1f}MB, may exceed OpenAI 50MB limit")
            
            # Check if it's already in a supported format
            supported_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
            file_ext = image_path.suffix.lower()
            
            if file_ext in supported_extensions:
                # File is already in supported format, just encode it
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                logger.info(f"Using original {file_ext} format for {image_path}")
            else:
                # Convert to PNG format using PIL
                logger.info(f"Converting {file_ext} to PNG format for OpenAI compatibility")
                with Image.open(image_path) as img:
                    # Convert to RGB if necessary (for formats like TIFF)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        # Convert RGBA/LA/P to RGB with white background
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save as PNG to bytes
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='PNG', optimize=True)
                    image_data = img_bytes.getvalue()
                    
                    # Check converted size
                    converted_size_mb = len(image_data) / (1024 * 1024)
                    logger.info(f"Converted image size: {converted_size_mb:.1f}MB")
                    
                    if converted_size_mb > 50:
                        # Resize if too large
                        logger.warning(f"Converted image too large ({converted_size_mb:.1f}MB), resizing...")
                        with Image.open(image_path) as img:
                            # Calculate new size to get under 50MB (rough estimation)
                            scale_factor = (45 / converted_size_mb) ** 0.5  # Square root for 2D scaling
                            new_width = int(img.width * scale_factor)
                            new_height = int(img.height * scale_factor)
                            
                            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            if img_resized.mode != 'RGB':
                                img_resized = img_resized.convert('RGB')
                            
                            img_bytes = io.BytesIO()
                            img_resized.save(img_bytes, format='PNG', optimize=True)
                            image_data = img_bytes.getvalue()
                            
                            final_size_mb = len(image_data) / (1024 * 1024)
                            logger.info(f"Resized image to {new_width}x{new_height}, final size: {final_size_mb:.1f}MB")
            
            # Encode to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            logger.info(f"Successfully encoded image {image_path} to base64 (size: {len(image_data)} bytes)")
            
            return base64_image
            
        except Exception as e:
            logger.error(f"Error converting image {image_path}: {e}")
            raise
    
    def analyze_flood_images(
        self, 
        flood_image_path: str, 
        satellite_image_path: str
    ) -> Dict[str, Any]:
        """
        Analyze flood overhead and satellite images using OpenAI GPT-4.1 Vision.
        Always uses the predefined flood analysis prompt.
        
        Args:
            flood_image_path: Path to flood depth visualization image
            satellite_image_path: Path to satellite/aerial image  
            
        Returns:
            Dictionary containing:
            - success: boolean
            - analysis: string with AI analysis
            - model: model used
            - tokens_used: number of tokens consumed
            - error: error message if failed
        """
        try:
            logger.info("Starting OpenAI flood image analysis...")
            
            # Convert and encode both images
            flood_b64 = self._convert_and_encode_image(flood_image_path)
            satellite_b64 = self._convert_and_encode_image(satellite_image_path)
            
            # Always use the predefined flood analysis prompt
            prompt = self._get_default_flood_analysis_prompt()
            
            # Prepare messages for OpenAI API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{flood_b64}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/png;base64,{satellite_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            # Make API call
            logger.info("Sending request to OpenAI GPT-4.1 Vision API...")
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                max_tokens=4000,
                temperature=0.1
            )
            
            analysis_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else None
            
            logger.info(f"OpenAI analysis completed successfully. Tokens used: {tokens_used}")
            
            return {
                "success": True,
                "analysis": analysis_text,
                "model": "gpt-4.1",
                "tokens_used": tokens_used
            }
            
        except Exception as e:
            error_msg = f"OpenAI analysis failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg
            }
    
    def analyze_single_image(
        self, 
        image_path: str, 
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single image using OpenAI GPT-4.1 Vision.
        
        Args:
            image_path: Path to image file
            prompt: Analysis prompt. If None, uses generic image analysis prompt.
            
        Returns:
            Dictionary containing analysis results or error information
        """
        try:
            logger.info(f"Starting OpenAI single image analysis for: {image_path}")
            
            # Convert and encode image
            image_b64 = self._convert_and_encode_image(image_path)
            
            # Use default prompt if none provided
            if not prompt:
                prompt = "Please analyze this image and describe what you see in detail, including any notable features, objects, or patterns."
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            # Make API call
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                max_tokens=2000,
                temperature=0.1
            )
            
            analysis_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else None
            
            logger.info(f"Single image analysis completed. Tokens used: {tokens_used}")
            
            return {
                "success": True,
                "analysis": analysis_text,
                "model": "gpt-4.1", 
                "tokens_used": tokens_used
            }
            
        except Exception as e:
            error_msg = f"Single image analysis failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg
            }
    
    def _get_default_flood_analysis_prompt(self) -> str:
        """
        Get the default comprehensive flood analysis prompt.
        
        Returns:
            Detailed prompt for flood damage analysis
        """
        return """Please analyze these two images for flood damage mitigation opportunities:

1. The first image shows a flood depth visualization with colored areas representing different flood depths (darker blue = shallow water, yellow/green = deeper water)
2. The second image shows a satellite/aerial view of the same area with actual terrain and development

First, roughly estimate the depth damage to the property and replacement cost from a 100 year flood.
Then, the images are a 100 year flood depth map, and an overhead satellite image of a property in Boulder, CO. Suggest specific, detailed, surgical interventions that could reduce 100 year flood risk to this property. Estimate the cost of each intervention, and consider regulatory barriers (like HOA requirements or FEMA flood defense suggestions). Rank the interventions with the cheapest approaches first, and estimate the corresponding loss in risk. Do not use markdown formatting."""