## Climate Resilience as a product 🌍

This project utilizes **LiDAR** data 🛰️ and **orthophotos** 📸 to create 3D models of environments. The primary goal is to identify and analyze features that can be modified to mitigate climate-related risks such as flooding, heat islands, or wildfire spread.

## The Goal 🎯

Our aim is to pinpoint specific areas or objects within a 3D environment (e.g., vegetation, buildings, infrastructure) that can be modified or managed to reduce climate-related risks.

For example, we could:
*   Identify vegetation likely to be a vector for fire spread to buildings.
*   Find optimal placements for floodwalls or other flood mitigation measures.
*   Detect rooftops suitable for green roofs 🌿 or solar panel installations ☀️.
*   Assess infrastructure vulnerabilities that need reinforcement.

## Technologies 🛠️

*   **FastAPI:** For building the web API that serves as the main interface to the system.
*   **Uvicorn:** As the ASGI server for running the FastAPI application.
*   **LiDAR:** For creating accurate 3D point clouds of the environment.
*   **Orthophotos:** Georeferenced aerial imagery providing color and texture information.
*   **Pytest:** For running automated tests.

## Codebase Organization 📂

The project is structured to separate concerns and make navigation and development easier:

*   **`app.py`**: The main entry point for the FastAPI application. It initializes the app, mounts routers, and defines background task processing.
*   **`routers/`**: Contains API endpoint definitions. Each file typically groups related endpoints (e.g., `jobs.py` for job status and download, `analysis.py` for analysis tasks).
    *   `health.py`: Basic health check endpoint.
    *   `jobs.py`: Endpoints for initiating processing, checking job status, and downloading results.
    *   `images.py`: Endpoints related to orthophoto retrieval.
    *   `analysis.py`: Endpoints for more specific analyses (e.g., flood risk - conceptual).
    *   `shared.py`: Utility functions or models shared across routers.
*   **`services/`**: Core logic and business operations. This is where the heavy lifting happens.
    *   `core/`: Essential services like geocoding (`geocode.py`) and point cloud processing/colorization (`process_point_cloud.py`).
    *   `data/`: Modules for fetching external data, such as LiDAR point clouds (`get_point_cloud.py`), orthophotos (`get_orthophoto.py`), and potentially other datasets like FEMA flood maps (`get_fema_risk.py`).
    *   `processing/`: Modules for more advanced data manipulation, such as point cloud I/O, orthophoto I/O, and potentially future analysis steps.
    *   `utils/`: Utility functions and classes used across different services (e.g., bounding box calculations, file handling, CRS transformations in `utils.py`, flood depth analysis in `flood_depth.py`).
    *   `visualization/`: Services related to generating reports or visual outputs (e.g., `summary_reporter.py`).
    *   `ai/`: (Likely for future AI/ML model integration for semantic segmentation or other analyses).
*   **`tests/`**: Contains all automated tests. The structure mirrors the main codebase (e.g., `test_api.py` for API tests, `test_core_services.py` for core service tests).
    *   `conftest.py`: Pytest fixtures and configuration.
    *   `test_dev_server.py`: Integration tests that run against a live (but locally stubbed) version of the FastAPI application.
*   **`data/`**: Directory for storing persistent data.
    *   `outputs/`: Default location for processed files (e.g., colorized point clouds).
    *   `orthophotos/`: Storage for downloaded orthophotos.
    *   `hazus/`, `spatial_index.json`: Related to specific data sources or indexing.
*   **`requirements.txt`**: Lists Python dependencies for the project.
*   **`README.md`**: This file! Provides an overview of the project.

## Workflow Overview 🗺️

The current primary workflow involves the following steps, orchestrated via the FastAPI application:

1.  **API Request:** A user (or another service) sends a request to the `/process` endpoint, typically providing an address.
    ```
    POST /process
    {
      "address": "123 Main St"
    }
    ```
2.  **Job Creation:** The application creates a unique job ID and initiates a background task.
3.  **Geocoding:** The address is geocoded to obtain latitude and longitude coordinates (see `services.core.geocode.Geocoder`).
4.  **Data Acquisition (Conceptual/Stubbed in tests):**
    *   A bounding box is generated around the coordinates.
    *   Relevant LiDAR data and orthophotos are identified and fetched (see `services.data.get_point_cloud.PointCloudDatasetFinder` and `services.data.get_orthophoto.NAIPFetcher`).
5.  **Point Cloud Processing & Colorization:**
    *   The LiDAR point cloud is processed.
    *   If orthophotos are available, the point cloud is colorized using the imagery (see `services.core.process_point_cloud.PointCloudProcessor`).
6.  **Output Storage:** The resulting processed file (e.g., a `.laz` file) is saved to the `data/outputs/` directory, named with the job ID.
7.  **Job Status Update:** The job status is updated to "completed," and the path to the output file is recorded.
8.  **Result Retrieval:** The user can poll the `/job/{job_id}` endpoint to check the status and, once completed, download the output file using the `/download/{job_id}` endpoint.

```
[User Request (Address) 📬] --> [API (/process)] --> [Background Job ⚙️] --> [Geocoding 🌍] --> [Data Fetching (LiDAR 🛰️, Orthophoto 📸)] --> [Point Cloud Colorization 🎨] --> [Output Saved 💾] --> [User Downloads Result 📥]
```

Future enhancements may include 3D semantic segmentation, physics-based simulations, and more detailed risk analysis.

## Project Setup and Execution

### Prerequisites

- Python 3.10 or higher
- Git

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd photogrammetry
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application 🚀

The application is a FastAPI web server.

1.  **Ensure your virtual environment is activated.** ✅
2.  **Start the FastAPI application:**
    ```bash
    python app.py
    ```
    This starts the Uvicorn server, typically on `http://localhost:8000`. The server will automatically reload on code changes in key directories.

    Alternatively, run with Uvicorn directly for more control:
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload --reload-dirs ./services,./routers
    ```

3.  **Access the API:**
    *   **API Documentation (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
    *   **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

### Running Tests 🧪

The project uses `pytest` for testing.

1.  **Ensure your virtual environment is activated.** ✅
2.  **Run all tests:**
    ```bash
    pytest
    ```
    This command discovers and runs all tests in the `tests/` directory.
    The test suite includes:
    *   Unit tests for individual modules and functions.
    *   Integration tests for service interactions.
    *   API tests that interact with the FastAPI endpoints (using `requests`).
    *   Development server tests (`tests/test_dev_server.py`) which run the application with stubbed external dependencies to test the overall flow.

3.  **Run specific test files or tests:**
    ```bash
    pytest tests/test_api.py  # Run a specific file
    pytest -k "test_process_flow"  # Run tests with names containing "test_process_flow"
    ```
    As per your instructions, regression tests using `test_dev_server.py` are included in the default `pytest` run.

---

Fighting climate change, one point cloud at a time! ✨🌍🌳
