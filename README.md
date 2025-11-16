# 🌌 Probabilistic Flappy Burd

A quantum-inspired Flappy Bird game using Energy-Based Modeling concepts from [THRML](https://docs.thrml.ai/).

![Probabilistic Flappy Burd](https://img.shields.io/badge/quantum-burd-yellow) ![THRML](https://img.shields.io/badge/THRML-inspired-orange)

## 🎮 Play It Now!

**[Play the game here!](https://scootykinz.github.io/probabilistic-flappy-burd/)**

## ✨ Features

### 🌟 Quantum Mode
- See **30 sampled future trajectories** in real-time
- Probability cloud visualization showing where the bird might go
- Energy-based sampling using Boltzmann distribution
- Heatmap showing probability density

### 🤖 AI Autoplay
- Watch an AI play using probabilistic reasoning
- AI makes decisions based on energy functions and trajectory predictions
- Toggle on/off to see the difference

### 📊 Progressive Difficulty
- First 5 pipes: Tutorial mode with huge gaps
- Score 5-10: Normal difficulty
- Score 10+: Gradually increasing challenge

## 🔬 The Science Behind It

This game demonstrates concepts from **Energy-Based Models (EBMs)** and **probabilistic graphical models**:

### Energy Function
Lower energy = more probable states. The energy function considers:
- **Gravity term**: Falling is natural (low energy when moving down)
- **Boundary penalties**: High energy near floor/ceiling
- **Pipe collision penalties**: Massive energy spike for collisions
- **Gap rewards**: Negative energy for threading pipe gaps

### Gibbs Sampling
At each timestep, the sampler:
1. Generates candidate future states (flap, fall, or random perturbation)
2. Calculates energy for each candidate
3. Samples according to **Boltzmann distribution**: `P(state) ∝ exp(-energy / temperature)`
4. Lower energy states are more likely to be selected

This creates a probability distribution over possible bird trajectories!

## 🛠️ Technical Details

### Built With
- **Vanilla JavaScript** - No frameworks, pure canvas rendering
- **THRML-inspired algorithms** - Energy-based sampling and Gibbs sampling
- **Python/THRML backend** (optional) - Real THRML library integration

### THRML Integration

This project was built for a THRML hackathon. While the main game runs pure JavaScript for browser compatibility, we also created a Python backend using the actual THRML library:

```python
from thrml import SpinNode, Block, SamplingSchedule
from thrml.models import IsingEBM, IsingSamplingProgram

# Create Ising model for bird trajectory
nodes = [SpinNode() for _ in range(TIME_STEPS)]
edges = [(nodes[i], nodes[i+1]) for i in range(TIME_STEPS - 1)]

# Energy biases (gravity, pipe avoidance)
biases = jnp.ones(TIME_STEPS) * gravity_bias

model = IsingEBM(nodes, edges, biases, weights, beta)
```

See `thrml_server.py` for the full implementation.

## 🚀 How to Run

### Play in Browser (Easiest)
Just open `index.html` in any modern browser!

### With Python Backend (Optional)
```bash
# Install dependencies
pip install thrml flask flask-cors

# Run the server
python thrml_server.py

# The game will connect to localhost:5000 for THRML-powered predictions
```

### Deploy to GitHub Pages
1. Push this repo to GitHub
2. Go to Settings → Pages
3. Select main branch as source
4. Your game will be live at `https://YOUR_USERNAME.github.io/REPO_NAME/`

## 🎯 Controls

- **Click** or **Space** to flap
- **Quantum Mode button** - Toggle probability cloud visualization
- **Autoplay button** - Let the AI play

## 📚 Learning Resources

- [THRML Documentation](https://docs.thrml.ai/)
- [Energy-Based Models](https://en.wikipedia.org/wiki/Energy-based_model)
- [Gibbs Sampling](https://en.wikipedia.org/wiki/Gibbs_sampling)
- [Boltzmann Distribution](https://en.wikipedia.org/wiki/Boltzmann_distribution)

## 🏆 Hackathon Project

Built for the THRML Hackathon to demonstrate:
- Understanding of energy-based models
- Probabilistic graphical model concepts
- Creative application of sampling algorithms
- Integration of THRML library

## 📝 Project Structure

```
.
├── index.html              # Main game (THRML-inspired JS implementation)
├── thrml_server.py         # Python backend with actual THRML library
├── README.md               # This file
└── assets/                 # (Optional) Game assets
```

## 🤝 Contributing

Feel free to fork and improve! Some ideas:
- Better AI using reinforcement learning
- More complex energy functions
- Multiplayer mode with entangled birds
- 3D visualization of probability space

## 📜 License

MIT License - Feel free to use for learning and fun!

## 🙏 Acknowledgments

- **Extropic AI** for creating THRML
- **THRML Documentation** for excellent examples
- The original Flappy Bird for inspiration
- Coffee ☕

---

**Made with 🌌 quantum mechanics and 🐦 birb energy**
