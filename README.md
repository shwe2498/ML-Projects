# Spotify Analytics

This project calculates the active user penetration rate for Spotify users in different countries based on specific criteria.

## Project Description

The script calculates the active user penetration rate for each country based on the following criteria:
- Last active date within the last 30 days
- At least 5 sessions
- At least 10 listening hours

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/spotify-analytics.git
    cd spotify-analytics
    ```
2. Ensure you have Python installed. You can create a virtual environment and install necessary packages using:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install pandas
    ```

## Usage

1. Add your data to the `data` dictionary in the `spotify_penetration_rate.py` file.
2. Run the script:
    ```sh
    python spotify_penetration_rate.py
    ```

## Output

The output will display the active user penetration rate for each country, rounded to 2 decimal places.

## License

This project is licensed under the MIT License.

