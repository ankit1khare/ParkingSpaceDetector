# Parking Space Detector - Tutorial and Steps

This guide explains how to create a Parking Space Detector using Vision Agent and Streamlit. The app allows you to count and mark empty parking spaces in an uploaded parking area image, and store the coordinates of the empty spaces.

## Steps to Build the Parking Space Detector

### 1. Prompting Vision Agent

I prompted Vision Agent with the following request:

"Count and mark the empty spaces and also store the coordinates of empty spaces."

Vision Agent provided a Python function that processes an image to detect and mark empty parking spaces while also storing their coordinates.

![image](https://github.com/user-attachments/assets/fb285e6b-b341-4d7f-85b9-a3595d908aef)


### 2. Installing Required Libraries

To run the provided function and build the app, I installed the following libraries locally as they are used in the provided code.

```
pip install streamlit pillow_heif vision-agent typing numpy PIL
```

### 3. Creating the Streamlit App

Using the function, I created a Streamlit app to provide a user-friendly interface for uploading images and viewing results. The app was saved in a file named app.py.

#### Here’s a summary of what the app does:

- Allows users to upload an image of a parking area.
- Processes the image using the provided function to detect empty parking spaces.
- Displays the marked image and lists the coordinates of the empty spaces.
- Provides download links for the marked image and the YAML file containing the coordinates.

### 4. Running the App

To run the app, execute the following command in the terminal:

```
streamlit run app.py
```

This will launch the app in your default web browser.


## File Structure

Here is the structure of the project:
```
.
├── empty_spaces.yaml         # example coordinates file
├── outputCode.py             # Contains the function for detecting empty parking spaces
├── marked_empty_spaces.pn    # example output file
└── app.py                    # Streamlit app file
```

## Examples

Input: A parking area image.<br>
Output: A marked image with empty spaces highlighted and the coordinates of the empty spaces.

![image](https://github.com/user-attachments/assets/b405ecd3-fbde-4785-9005-f00900dc2c7e)
![image](https://github.com/user-attachments/assets/42279038-ddc3-488a-9fd7-49239e72d38d)



