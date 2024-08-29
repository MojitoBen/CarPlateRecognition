# Smart Pillar

## Purpose
The Smart Pillar combines license plate recognition with RFID receiver technology to detect vehicles on the road. The identified data is then matched and stored in a cloud database.    
<img src="https://github.com/MojitoBen/CarPlateRecognition/blob/main/pillar/images/device.png" width="350">

## Project structure

<img src="https://github.com/MojitoBen/CarPlateRecognition/blob/main/pillar/images/flow4.png" width="500">

- **Systems Used**:
  - License Plate Recognition System
  - RFID Detection System

- **Data Obtained**:
  - License plate numbers
  - eTag data

- **Data Storage**:
  - All information is stored in a database.

- A unique correspondence is established between each license plate and its corresponding eTag data.
- This process effectively registers the vehicle's identity.
    
## Result

#### Detect 
<img src="https://github.com/MojitoBen/CarPlateRecognition/blob/main/pillar/images/2024-07-10_17.30.29_test.png" width="350"><img src="https://github.com/MojitoBen/CarPlateRecognition/blob/main/pillar/images/2024-07-10_17.32.29_test.png" width="350">

#### Etags & Database Records

<img src="https://github.com/MojitoBen/CarPlateRecognition/blob/main/pillar/images/db_carplate_records.png" width="350"><img src="https://github.com/MojitoBen/CarPlateRecognition/blob/main/pillar/images/db_rfid_records.png" width="350">
<img src="https://github.com/MojitoBen/CarPlateRecognition/blob/main/pillar/images/db_duplicate_bindings.png" width="350">
<img src="https://github.com/MojitoBen/CarPlateRecognition/blob/main/pillar/images/db_duplicate_groups.png" width="350">
<img src="https://github.com/MojitoBen/CarPlateRecognition/blob/main/pillar/images/db_binding.png" width="350">
