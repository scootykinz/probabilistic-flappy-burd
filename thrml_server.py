from flask import Flask, request, jsonify
from flask_cors import CORS
import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for browser requests

# Constants matching our JavaScript game
GRID_HEIGHT = 600  # Canvas height
HEIGHT_BINS = 20   # Discretize height into 20 bins  
TIME_STEPS = 15    # Predict 15 frames into future (simplified)

def create_bird_trajectory_model(current_y, current_velocity, pipes):
    """
    Create a THRML Ising model for bird trajectory prediction
    Uses spin nodes (-1/+1) to represent up/down movement tendencies
    
    Args:
        current_y: Current bird Y position (0-600)
        current_velocity: Current bird velocity
        pipes: List of pipe positions and gaps
    
    Returns:
        Sampled trajectories
    """
    
    # Create a chain of spin nodes representing movement decisions over time
    # +1 = tendency to go up, -1 = tendency to go down
    nodes = [SpinNode() for _ in range(TIME_STEPS)]
    
    # Create temporal edges (each decision influences the next)
    edges = [(nodes[i], nodes[i+1]) for i in range(TIME_STEPS - 1)]
    
    # Biases: gravity bias (encourages downward movement)
    gravity_bias = 0.5 if current_velocity > 0 else -0.3
    biases = jnp.ones(TIME_STEPS) * gravity_bias
    
    # Add pipe avoidance biases
    for pipe in pipes:
        pipe_x = pipe['x']
        pipe_top = pipe['topHeight']
        pipe_bottom = pipe['bottomY']
        
        # Estimate which time step this pipe matters
        frames_until_pipe = int((pipe_x - 150) / 3)
        
        if 0 <= frames_until_pipe < TIME_STEPS:
            # If bird is too high (near top pipe), bias down
            if current_y < pipe_top + 50:
                biases = biases.at[frames_until_pipe].set(-1.0)  # Go down
            # If bird is too low (near bottom pipe), bias up  
            elif current_y > pipe_bottom - 50:
                biases = biases.at[frames_until_pipe].set(1.0)   # Go up
    
    # Weights: encourage smooth decisions (don't flip-flop too much)
    weights = jnp.ones(len(edges)) * 0.5
    
    # Temperature
    beta = jnp.array(1.0)
    
    # Create Ising model
    model = IsingEBM(nodes, edges, biases, weights, beta)
    
    # Sampling program
    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    
    # Sample
    key = jax.random.key(42)
    k_init, k_samp = jax.random.split(key, 2)
    init_state = hinton_init(k_init, model, free_blocks, ())
    
    schedule = SamplingSchedule(n_warmup=50, n_samples=50, steps_per_sample=2)
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    
    return samples, current_y

@app.route('/predict', methods=['POST'])
def predict_trajectory():
    """
    API endpoint to predict bird trajectory using THRML Ising model
    
    Expects JSON:
    {
        "birdY": 300,
        "velocity": 2.5,
        "pipes": [{"x": 400, "topHeight": 200, "bottomY": 400}, ...]
    }
    
    Returns:
    {
        "trajectories": [[y0, y1, ..., y14], ...],  # Multiple sampled Y positions
        "heatmap": [prob0, prob1, ..., prob19],  # Probability for each height bin
        "method": "THRML"
    }
    """
    try:
        data = request.json
        bird_y = data['birdY']
        velocity = data['velocity']
        pipes = data['pipes']
        
        # Get samples from THRML Ising model
        samples, start_y = create_bird_trajectory_model(bird_y, velocity, pipes)
        
        # Convert spin samples to Y positions
        # Spin: +1 means go up, -1 means go down
        trajectories = []
        
        for sample in samples:
            trajectory = [start_y]
            current_y = start_y
            current_v = velocity
            
            # Convert each spin decision to a Y position
            for t in range(TIME_STEPS):
                spin = sample[t]  # +1 or -1
                
                # If spin is +1, add upward impulse; if -1, let gravity dominate
                if spin > 0:
                    current_v = -6.5  # Flap!
                
                # Apply physics
                current_v += 0.25  # Gravity
                current_v = min(current_v, 8)  # Terminal velocity
                current_y += current_v
                
                # Clamp to screen
                current_y = max(0, min(GRID_HEIGHT, current_y))
                
                trajectory.append(current_y)
            
            trajectories.append(trajectory[:10])  # First 10 steps
        
        # Create heatmap from trajectories
        heatmap = np.zeros(HEIGHT_BINS)
        for traj in trajectories[:20]:
            for y in traj[:5]:  # Look at next 5 frames
                bin_idx = int((y / GRID_HEIGHT) * HEIGHT_BINS)
                bin_idx = max(0, min(HEIGHT_BINS - 1, bin_idx))
                heatmap[bin_idx] += 1
        
        heatmap = (heatmap / heatmap.sum()).tolist() if heatmap.sum() > 0 else heatmap.tolist()
        
        return jsonify({
            'status': 'success',
            'trajectories': trajectories[:20],
            'heatmap': heatmap,
            'method': 'THRML-Ising'
        })
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e),
            'method': 'fallback'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'THRML server running!'})

if __name__ == '__main__':
    print("🚀 Starting THRML Flappy Bird Server...")
    print("📡 Server running on http://localhost:5001")
    print("🐦 Ready to predict bird trajectories!")
    app.run(host='0.0.0.0', port=5001, debug=True)
