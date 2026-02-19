# Car License Plate Recognition System

### Flow Process
1. Detect the Car use Pre-trian Model (Car Detection Model) and get the Car Bounding Box.
2. Crop the Car Image from the original image by using the Car Bounding Box.
3. Use the cropped Car Image to detect the License Plate by using another Pre-train Model (License Plate Detection Model) and get the License Plate Bounding Box.
4. Crop the License Plate Image from the Car Image by using the License Plate Bounding Box.
5. Use the cropped License Plate Image to recognize the characters on the License Plate by using another Pre-train Model (License Plate Recognition Model) and get the License Plate Number.




### Thai Font
1. Install Thai font support on your system to ensure proper rendering of Thai characters. This may involve downloading and installing a Thai font package.

    ```commandline
    sudo apt update
    sudo apt install fonts-thai-tlwg
    ```
   
2. Use "Loma" or "Garuda" (Alternatives to Tahoma):
Once installed, the fonts are located at /usr/share/fonts/truetype/tlwg/.

    Loma.ttf: Very similar to Tahoma (User Interface font).

    Garuda.ttf: Very clear, bold, good for OSD (On Screen Display).

    Kinnari.ttf: Serif font (like Angsana).
