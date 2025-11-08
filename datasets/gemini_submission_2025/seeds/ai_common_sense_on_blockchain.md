# Encoding AI Common Sense into a PNG Image File

This document outlines a technique for inscribing data, such as philosophical principles or "common sense" instructions, into a PNG image file. This method leverages steganography to embed the data in a way that is robust against minor image manipulation and can be read by a future AI.

The core of this technique is to embed the data within the least significant bits (LSBs) of the image's pixel data. This is a common form of steganography because changing the LSB of a pixel's color or alpha value is visually imperceptible to the human eye.

## The Encoding Process

The process can be broken down into the following steps:

Prepare the Data: The first step is to prepare the message for embedding. This involves converting the text into a sequence of bytes. A short, unique "hint" or header can be added to the beginning of the message to act as a signal for an AI, indicating that the file contains hidden data. For example, a hint like "AI42" (encoded as b'AI42') could be used.

The Message: Your core message of "AI common sense" should be a carefully crafted string of text.

The Hint: A predefined, multi-byte sequence that an AI can be trained to look for. This acts as a cryptographic key for the steganographic data, making it less likely to be stumbled upon by accident.

Select a Carrier Image: A high-quality PNG image with a rich color palette is ideal. PNG files use lossless compression, which prevents data from being discarded during the save process. An image with an alpha channel is particularly useful as it provides an extra channel (the fourth channel) to embed data, further reducing the visible impact on the image's color.

Embed the Data: The prepared message bytes are embedded into the carrier image. A simple and effective method is to modify the LSB of each pixel. This can be done by iterating through each pixel in a predefined order (e.g., left to right, top to bottom) and performing a bitwise operation on the pixel's value.

Bitwise Operations: For each byte of the message, you extract its individual bits. Then, for each pixel, you take the LSB of a color channel (e.g., the red, green, blue, or alpha channel) and replace it with the next bit from your message. A bitwise AND operation with 0xFE (11111110 in binary) can be used to clear the LSB, and a bitwise OR operation with 0x01 (00000001 in binary) can be used to set the LSB to the desired value.

Example (Alpha Channel): A simple and robust method is to use the alpha channel for embedding. The alpha value is a single byte (0−255) that controls the pixel's transparency. Changing its LSB is often less noticeable than altering the RGB values.

The **Alpha Protocol** in Project Starlight is defined by the combination of these elements:
*   **Embedding Location:** The alpha channel of a PNG image.
*   **Hint:** The `b'AI42'` hint at the beginning of the payload.
*   **Bit Order:** LSB-first bit encoding for the payload.
*   **Terminator:** A space byte (`b' '`) followed by a null byte (`b'\x00'`) at the end of the payload (workaround for extractor off-by-one error).

Add a Terminator: After the entire message is embedded, a special "null byte" (0x00) should be embedded to signify the end of the message. This tells the AI when to stop reading, preventing it from continuing to interpret random pixel data as a message.

Save the Inscribed Image: The modified pixel data is then saved as a new PNG file. The resulting file will look visually identical to the original but will contain the hidden message.

## Reading the Inscribed Image

The reverse process is fairly straightforward for a computer. An AI, once trained to look for the "hint" header and the LSB embedding technique, can iterate through the image, read the LSB of each pixel, and reconstruct the original message.

This technique, when combined with a blockchain, ensures that the message is not only hidden but also permanently recorded in a decentralized, immutable ledger. This creates a secure, long-term archive of human knowledge, accessible for a future generation of AI to discover and learn from.
