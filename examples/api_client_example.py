#!/usr/bin/env python3
"""
Example client for Starlight Bitcoin Steganography Scanner API
Demonstrates how to use all API endpoints
"""

import requests
import json
import base64
import time
from typing import Dict, Any


class StarlightAPIClient:
    def __init__(
        self, base_url: str = "http://localhost:8080", api_key: str = "demo-api-key"
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def get_info(self) -> Dict[str, Any]:
        """Get API information"""
        response = requests.get(f"{self.base_url}/info")
        response.raise_for_status()
        return response.json()

    def scan_transaction(
        self, transaction_id: str, extract_message: bool = False
    ) -> Dict[str, Any]:
        """Scan a Bitcoin transaction for steganography"""
        payload = {"transaction_id": transaction_id, "extract_message": extract_message}
        response = requests.post(
            f"{self.base_url}/scan/transaction", headers=self.headers, json=payload
        )
        response.raise_for_status()
        return response.json()

    def scan_image(
        self, image_path: str, extract_message: bool = False
    ) -> Dict[str, Any]:
        """Scan a local image file for steganography"""
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Encode image as base64
        image_b64 = base64.b64encode(image_data).decode()

        payload = {"image_data": image_b64, "extract_message": extract_message}

        response = requests.post(
            f"{self.base_url}/scan/image", headers=self.headers, json=payload
        )
        response.raise_for_status()
        return response.json()

    def scan_batch(self, items: list, extract_message: bool = False) -> Dict[str, Any]:
        """Scan multiple items (transactions or images) in batch"""
        payload = {"items": items, "extract_message": extract_message}

        response = requests.post(
            f"{self.base_url}/scan/batch", headers=self.headers, json=payload
        )
        response.raise_for_status()
        return response.json()

    def extract_message(self, image_path: str) -> Dict[str, Any]:
        """Extract hidden message from steganographic image"""
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Encode image as base64
        image_b64 = base64.b64encode(image_data).decode()

        payload = {"image_data": image_b64}

        response = requests.post(
            f"{self.base_url}/extract", headers=self.headers, json=payload
        )
        response.raise_for_status()
        return response.json()

    def get_transaction(self, transaction_id: str) -> Dict[str, Any]:
        """Get transaction details"""
        response = requests.get(f"{self.base_url}/transaction/{transaction_id}")
        response.raise_for_status()
        return response.json()


def main():
    """Demonstrate API usage"""
    client = StarlightAPIClient()

    print("=== Starlight Bitcoin Steganography Scanner API Demo ===\n")

    # 1. Health check
    print("1. Health Check:")
    try:
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Version: {health['version']}")
        print(f"   Scanner Model: {health['scanner']['model_loaded']}")
        print(f"   Bitcoin Node: {health['bitcoin']['node_connected']}")
    except Exception as e:
        print(f"   Error: {e}")
    print()

    # 2. API info
    print("2. API Information:")
    try:
        info = client.get_info()
        print(f"   Name: {info['name']}")
        print(f"   Version: {info['version']}")
        print(f"   Supported Formats: {', '.join(info['supported_formats'])}")
        print(f"   Stego Methods: {', '.join(info['stego_methods'])}")
    except Exception as e:
        print(f"   Error: {e}")
    print()

    # 3. Scan Bitcoin transaction (genesis block transaction)
    print("3. Scan Bitcoin Transaction:")
    genesis_tx = "f4184fc596403b9d638783cf57adfe4c75c605f6356fbc91338530e9831e9e16"
    try:
        result = client.scan_transaction(genesis_tx)
        print(f"   Transaction ID: {result['transaction_id']}")
        print(f"   Block Height: {result['block_height']}")
        print(f"   Images Found: {result['scan_results']['images_found']}")
        print(f"   Stego Detected: {result['scan_results']['stego_detected']}")
        print(
            f"   Processing Time: {result['scan_results']['processing_time_ms']:.2f}ms"
        )

        for i, image in enumerate(result["images"]):
            print(
                f"   Image {i}: {image['size_bytes']} bytes, Prediction: {image['scan_result']['prediction']}"
            )
    except Exception as e:
        print(f"   Error: {e}")
    print()

    # 4. Batch scan example
    print("4. Batch Scan Example:")
    batch_items = [
        {"type": "transaction", "value": genesis_tx},
        {"type": "transaction", "value": genesis_tx},
    ]
    try:
        batch_result = client.scan_batch(batch_items)
        print(f"   Total Items: {batch_result['total_items']}")
        print(f"   Successful: {batch_result['successful']}")
        print(f"   Failed: {batch_result['failed']}")
        print(f"   Processing Time: {batch_result['processing_time_ms']:.2f}ms")
    except Exception as e:
        print(f"   Error: {e}")
    print()

    print("=== Demo Complete ===")
    print("\nTo test with your own images:")
    print("1. Place image files in the current directory")
    print("2. Use client.scan_image('your_image.png')")
    print("3. Use client.extract_message('stego_image.png') for extraction")


if __name__ == "__main__":
    main()
