from flask import Flask, render_template, jsonify, request
from KMeans import KMeans  # Ensure your KMeans class is correctly imported
import numpy as np
import matplotlib
import base64
import io

app = Flask(__name__)
matplotlib.use('Agg')

points = np.empty((0, 2))

@app.route('/')
def index():
    global points
    points = np.random.uniform(-10, 10, (300, 2))  # Generate points on page load
    return render_template('index.html')

@app.route('/generate-dataset', methods=['POST'])
def generate():
    global points
    points = np.random.uniform(-10, 10, (300, 2))
    return jsonify({'points': points.tolist()})

@app.route('/find-kmeans', methods=['POST'])
def find_kmeans():
    k = int(request.json.get('k'))
    method = request.json.get('method')

    if method == 'manual':
        centers = request.json.get('centers') 
    else:
        centers = []

    kmeans = KMeans(points, k, method, centers)
    
    kmeans.lloyds()

    if kmeans.snaps:
        gif_bytes = io.BytesIO()
        kmeans.snaps[0].save(
            gif_bytes,
            format='GIF',
            optimize=False,
            save_all=True,
            append_images=kmeans.snaps[1:],
            duration=500
        )
        gif_bytes.seek(0)
        plot_url = base64.b64encode(gif_bytes.getvalue()).decode('utf8')
    else:
        return jsonify({'error': 'No frames captured'}), 500

    return jsonify({'plot_url': plot_url})

@app.route('/run-convergence', methods=['POST'])
def run_convergence():
    k = int(request.json.get('k'))
    method = request.json.get('method')
    
    if method == 'manual':
        centers = request.json.get('centers') 
    else:
        centers = []
        
    kmeans = KMeans(points, k, method, centers)
    kmeans.lloyds()  # Run Lloyd's algorithm
    
    # Take the last snapshot as the image
    if kmeans.snaps:
        last_snap = kmeans.snaps[-1]  # Get the last frame
        # Convert the last frame to bytes
        img_byte_arr = io.BytesIO()
        last_snap.save(img_byte_arr, format='PNG')  # Change format if necessary
        img_byte_arr.seek(0)
        plot_url = base64.b64encode(img_byte_arr.getvalue()).decode('utf8')
        return jsonify({'plot_url': plot_url})
    else:
        return jsonify({'error': 'No frames captured'}), 500

@app.route('/reset', methods=['POST'])
def reset():
    global points
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    app.run(debug=True, port=3000)

