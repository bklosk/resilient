
🌍 Climate Resilience Intelligence

This repo contains code that processes geographic data, identifies vulnerabilities to climate risks, and recommends improvements.

🎯 What We’re Solving

We use technology to:
	•	Identify vegetation near buildings that might spread wildfires.
	•	Spot ideal locations for flood barriers.
	•	Recommend rooftops for solar panels or green roofs.
	•	Suggest infrastructure improvements to withstand extreme weather.

🛠️ Tech Stack

Here’s what we’re using and why:
	•	FastAPI: Easy, fast web API framework in Python.
	•	LiDAR: Laser-based elevation data for detailed terrain mapping.
	•	Orthophotos: High-quality aerial images for visual accuracy.
	•	WRTC / HAZUS: High-quality climate risk data from the US Government

📂 Repo Structure

Here’s what’s in the repository:
	•	app.py: Entry point for the FastAPI app. Starts everything up.
	•	routers/: Routes requests coming from users.
	•	jobs.py: Manages tasks in the background.
	•	analysis.py: Runs climate vulnerability analysis.
	•	services/: Handles the heavy lifting.
	•	core/: Processes 3D point cloud data.
	•	data/: Fetches LiDAR and orthophoto data.
	•	processing/: Crunches data into usable formats.
	•	utils/: Handy utility functions used throughout the project.
	•	tests/: Tests to make sure everything runs smoothly.
	•	data/: Stores all data files.
	•	outputs/: Finished models and processed data.
	•	orthophotos/: Raw aerial images.
	•	deployment/: Docker files for easy deployment.
	•	requirements.txt: All Python libraries needed.
	•	README.md: This file!

🚀 How the App Works

Here’s the typical workflow:
	1.	A user submits an address via our API.
	2.	The app creates a new job.
	3.	It finds the geographic coordinates for the address.
	4.	It gathers LiDAR and aerial images for the location.
	5.	Processes that data to create a detailed 3D model.
	6.	Saves the model to storage.
	7.	The user can then download the finished 3D model.

We’re continually enhancing this process, like adding smarter analysis powered by AI models.

🛠️ How to Run the App (Setup)

Follow these steps to get the app running locally:

Prerequisites:
	•	Python 3.10+
	•	Git

Steps:
	1.	Clone this repo:

git clone <repository-url>
cd photogrammetry

	2.	Create and activate a Python virtual environment:

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

	3.	Install dependencies:

pip install -r requirements.txt

	4.	Start the application:

uvicorn app:app --reload

Visit http://127.0.0.1:8000 to use the API.

⸻

Changing the world, one home at a time 🌍✨