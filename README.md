## This is an application that uses the SSD MobileNet V2 FPNLite prediction model for object detection.
### Tech Stack
<ul>
<li>Python3.8</li>
<li>TensorFlow</li>
<li>COCO Labels</li>
</ul>

## Creating and activating virtual environment
<ol>
<li> $python3.8 -m venv detector</li>
<li> $source detector/bin/activate</li>
<li> $pip install -r requirements.txt</li>
</ol>

## Running the application
* There is an images folder that you can save additional images to. To run the object detection:
<ol>
    <li> on the line image_path replace the path with "./images/(your new image name.jpg)</li>
    <li> (from the root folder) python3.8 object_detector.py</li>
</ol>