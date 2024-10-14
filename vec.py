import os
import tempfile
import cv2
import numpy as np
from flask import Flask, request, send_file, render_template, after_this_request
from werkzeug.utils import secure_filename
from scipy.interpolate import splprep, splev

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def simplify_contour(contour, epsilon=0.01):
    return cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)

def smooth_contour(contour, smoothing=10):
    x, y = contour.T
    x = x.tolist()[0]
    y = y.tolist()[0]
    
    # Pastikan jumlah titik cukup untuk interpolasi
    if len(x) < 4:
        return contour

    tck, u = splprep([x, y], u=None, s=smoothing, per=1, k=min(3, len(x)-1))
    u_new = np.linspace(u.min(), u.max(), max(len(x), 1000))
    x_new, y_new = splev(u_new, tck, der=0)
    
    res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
    return np.array(res_array, dtype=np.int32)

def bitmap_to_vector(input_path, output_path):
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = img.shape[:2]
    svg_content = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    
    for contour in contours:
        if len(contour) < 4:
            continue  # Lewati kontur yang terlalu pendek
        
        # Simplify contour
        simplified = simplify_contour(contour)
        
        # Smooth contour
        try:
            smoothed = smooth_contour(simplified)
        except Exception:
            smoothed = simplified  # Gunakan kontur yang disederhanakan jika penghalusan gagal
        
        # Convert to SVG path
        path = "M"
        for point in smoothed:
            x, y = point[0]
            path += f" {x},{y}"
        path += "Z"
        
        svg_content += f'<path d="{path}" fill="none" stroke="black" />'
    
    svg_content += '</svg>'
    
    with open(output_path, 'w') as f:
        f.write(svg_content)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            temp_dir = tempfile.mkdtemp()
            try:
                input_path = os.path.join(temp_dir, filename)
                file.save(input_path)
                output_path = os.path.join(temp_dir, 'output.svg')
                bitmap_to_vector(input_path, output_path)
                
                @after_this_request
                def remove_file(response):
                    try:
                        os.remove(input_path)
                        os.remove(output_path)
                        os.rmdir(temp_dir)
                    except Exception as error:
                        app.logger.error("Error removing or closing downloaded file handle", error)
                    return response
                
                return send_file(output_path, as_attachment=True, download_name='output.svg')
            except Exception as e:
                return str(e)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)