#  Photogrammetree 🌳 
## Climate Disaster Mitigation 🌍

This project uses **LiDAR** data 🛰️ and **orthophotos** 📸 combined with **3D semantic segmentation** 🤖 and FEA/FDS to identify and analyze mutable candidates for climate risk mitigation.

## The Goal 🎯

Our aim is to pinpoint specific areas or objects within a 3D environment (e.g., vegetation, buildings, infrastructure) that can be modified or managed to reduce climate-related risks like flooding, heat islands, or wildfire spread.

For example, we could:
*   Identify vegetation likely to be a vector for fire spread to buildings.
*   Find optimal placements for floodwalls, sandbags, or home risers.
*   Detect rooftops suitable for conversion to steel roofs 🌿 or solar panel installations ☀️.
*   Assess infrastructure vulnerabilities that need reinforcement.

## Technologies 🛠️

*   **LiDAR:** For creating accurate 3D point clouds of the environment.
*   **Orthophotos:** Georeferenced aerial imagery providing color and texture information.
*   **3D Semantic Segmentation:** Machine learning models to classify objects within the 3D point cloud (e.g., ground, buildings, low vegetation, high vegetation).

## Workflow Overview (Conceptual) 🗺️

```
[LiDAR Data 🛰️ + Orthophotos 📸] --> [Preprocessing & Colorization 🎨] --> [3D Semantic Segmentation 🤖] --> [Candidate Identification 🎯] --> [Physics Simulation & ML Analysis 🌪️🔥🌊] --> [Mitigation Strategies & Visualization 📊]
```

1.  **Data Acquisition & Preprocessing:**
    *   Collect LiDAR scans 📡 and orthophotos 🖼️.
    *   Align and fuse these datasets.
    *   Colorize the LiDAR point cloud using orthophoto imagery (implemented in `services/core/process_point_cloud.py`).
2.  **3D Semantic Segmentation:**
    *   Train or use a pre-trained model to classify points in the cloud (e.g., using Open3D-ML) 🧠.
3.  **Candidate Identification & Analysis:**
    *   Develop algorithms to analyze the segmented point cloud.
    *   Identify "mutable" objects/areas based on their class and properties.
    *   Evaluate their potential for climate risk mitigation.
4.  **Physics-Based Simulation & ML Analysis 🌪️🔥🌊:**
    *   Run Finite Element Analysis (FEA) and Fire Dynamics Simulator (FDS) simulations on the segmented point cloud.
    *   Simulate physical phenomena like fires, floods, and high winds.
    *   Utilize Machine Learning to analyze simulation results and refine mitigation strategies.
5.  **Visualization & Reporting:**
    *   Visualize the results in 3D.
    *   Generate reports for decision-making.

---

## Project Setup and Execution

### Prerequisites

- Python 3.10 or higher
- Git (for cloning the repository, if applicable)

### Setup in a GitHub Codespace or Local Environment

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository-url>
    cd photogrammetry
    ```

2.  **Create and activate a Python virtual environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    (On Windows, use `venv\Scripts\activate`)

3.  **Install Python dependencies:**
    The required Python packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

### Running the Pipeline 🚀

The main entry point is now the FastAPI application defined in `app.py`. This application provides a web API for processing LiDAR point clouds with orthophotos and other related functionalities.

To run the pipeline:

1.  **Ensure your virtual environment is activated.** ✅
    ```bash
    source venv/bin/activate
    ```
    (On Windows, use `venv\Scripts\activate`)

2.  **Start the FastAPI application:**
    You can run the application directly using Python:
    ```bash
    python app.py
    ```
    This will start the Uvicorn server, typically on `http://localhost:8000`. The server will automatically reload if you make changes to files in the `services` or `routers` directories.

    Alternatively, you can run it directly with Uvicorn for more control:
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload --reload-dirs ./services --reload-dirs ./routers
    ```

3.  **Use the API endpoints or web interface to process data:**
    *   **API Documentation (Swagger UI):** Once the server is running, you can access the interactive API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).
    *   **OpenAPI Specification:** The OpenAPI JSON schema is available at [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json).
    *   **Health Check:** Check the API status at [http://localhost:8000/health](http://localhost:8000/health).
    *   **Processing Endpoint:** To initiate point cloud processing, send a POST request to the `/process` endpoint with an address (e.g., `{"address": "123 Main St"}`).

    The application will process the data in the background. You can check the status of a job using its job ID via the `/job/{job_id}` endpoint and download results using `/download/{job_id}`. Output files are typically saved in the `data/outputs/` directory.

### Running Tests

The project includes a comprehensive test suite. After installing dependencies (including development dependencies if separated in the future), you can run the tests using `pytest`.

1.  **Ensure your virtual environment is activated.** ✅
    ```bash
    source venv/bin/activate
    ```

2.  **Run all tests:**
    ```bash
    pytest
    ```
    This command will discover and run all tests in the `tests/` directory. The tests are designed to use lightweight stubs and mocks, so they execute quickly without requiring external data downloads or live API calls for most unit/integration tests.

3.  **Run specific test files or tests:**
    You can run specific test files:
    ```bash
    pytest tests/test_api.py
    ```
    Or specific tests by name:
    ```bash
    pytest -k test_process_flow
    ```

4.  **Regression Tests (Dev Server):**
    The file `tests/test_dev_server.py` contains tests that spin up a local development version of the FastAPI server with stubbed external services. These act as higher-level integration or regression tests for the API flow. They are included when you run `pytest`.

---

Fighting climate change, one point cloud at a time! ✨🌍🌳
