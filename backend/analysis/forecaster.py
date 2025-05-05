# backend/analysis/forecaster.py
# Logic for time series forecasting

import pandas as pd
import logging
from typing import Optional, Dict, Any, Tuple
import plotly.graph_objects as go
import plotly.express as px

# Attempt to import forecasting libraries
PROPHET_AVAILABLE = False
STATSMODELS_AVAILABLE = False
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly, plot_components_plotly
    PROPHET_AVAILABLE = True
except ImportError:
    logging.warning("Prophet library not found. Prophet forecasting disabled. `pip install prophet`")
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX # Example model
    from statsmodels.tsa.arima.model import ARIMA # Example model
    # from statsmodels.tsa.holtwinters import ExponentialSmoothing # Another option
    STATSMODELS_AVAILABLE = True
except ImportError:
    logging.warning("statsmodels library not found. ARIMA/SARIMAX forecasting disabled. `pip install statsmodels`")


logger = logging.getLogger(__name__)


def _prepare_time_series_df(df: pd.DataFrame, time_col: str, target_col: str, freq: Optional[str] = None) -> pd.DataFrame:
    """Prepares DataFrame for time series analysis (sets index, handles frequency)."""
    if time_col not in df.columns: raise KeyError(f"Time column '{time_col}' not found.")
    if target_col not in df.columns: raise KeyError(f"Target column '{target_col}' not found.")

    ts_df = df[[time_col, target_col]].copy()
    ts_df[time_col] = pd.to_datetime(ts_df[time_col])
    ts_df = ts_df.sort_values(by=time_col)
    ts_df = ts_df.set_index(time_col)

    # Handle potential duplicate timestamps (e.g., aggregate)
    if ts_df.index.has_duplicates:
        logger.warning(f"Duplicate timestamps found in '{time_col}'. Aggregating using mean.")
        ts_df = ts_df.groupby(ts_df.index).mean()

    # Infer frequency or resample if needed
    inferred_freq = pd.infer_freq(ts_df.index)
    target_freq = freq or inferred_freq or 'D' # Default to daily if cannot infer
    logger.info(f"Target frequency for time series: {target_freq} (Inferred: {inferred_freq})")

    # Resample to ensure regular frequency (important for some models like ARIMA)
    # Fill missing values after resampling (e.g., forward fill)
    ts_df = ts_df.asfreq(target_freq).fillna(method='ffill').fillna(method='bfill') # Fill forward then backward

    # Rename for Prophet compatibility if needed
    ts_df_prophet = ts_df.reset_index()
    ts_df_prophet.rename(columns={time_col: 'ds', target_col: 'y'}, inplace=True)

    return ts_df, ts_df_prophet # Return both original indexed and Prophet-formatted


def generate_forecast(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    periods: int,
    freq: Optional[str] = None, # e.g., 'D', 'W', 'M', 'Q', 'Y'
    model_name: str = 'prophet' # 'prophet', 'arima', 'sarimax', 'simple'
    ) -> Tuple[Optional[pd.DataFrame], Optional[go.Figure]]:
    """
    Generates a forecast using the specified model.

    Args:
        df: Input DataFrame with time and target columns.
        time_col: Name of the time/date column.
        target_col: Name of the column to forecast.
        periods: Number of periods into the future to forecast.
        freq: Optional frequency string (e.g., 'D', 'M'). If None, attempts to infer.
        model_name: The forecast model to use ('prophet', 'arima', 'sarimax', 'simple').

    Returns:
        A tuple containing:
        - forecast_df (pd.DataFrame): DataFrame with historical data and future predictions.
        - forecast_fig (go.Figure): Plotly figure visualizing the forecast.
        Returns (None, None) on failure.
    """
    logger.info(f"Generating {periods}-period forecast for '{target_col}' using model '{model_name}'.")
    forecast_df = None
    forecast_fig = None

    try:
        ts_df, ts_df_prophet = _prepare_time_series_df(df, time_col, target_col, freq)
        target_freq = ts_df.index.freqstr or 'D' # Get frequency used after prep

        # --- Model Selection & Forecasting ---
        if model_name.lower() == 'prophet':
            if not PROPHET_AVAILABLE: raise ImportError("Prophet model selected but library not installed.")
            logger.info("Fitting Prophet model...")
            # Basic Prophet model - add holidays, regressors etc. later
            model = Prophet()
            model.fit(ts_df_prophet)
            future = model.make_future_dataframe(periods=periods, freq=target_freq)
            forecast = model.predict(future)
            forecast_df = forecast # Includes history and prediction components
            logger.info("Prophet forecast generated.")
            # Create Plotly figure using Prophet's plotting functions
            fig1 = plot_plotly(model, forecast)
            fig2 = plot_components_plotly(model, forecast)
            # Combine or select one? For now, just return main forecast plot
            forecast_fig = fig1
            forecast_fig.update_layout(title=f"Prophet Forecast: {target_col}")


        elif model_name.lower() in ['arima', 'sarimax']:
            if not STATSMODELS_AVAILABLE: raise ImportError("statsmodels library not installed for ARIMA/SARIMAX.")
            logger.info(f"Fitting {model_name.upper()} model...")
            # Basic ARIMA/SARIMAX model - order (p,d,q) needs tuning (e.g., auto_arima)
            # Placeholder: Use simple order (1,1,1)
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 12) if model_name.lower() == 'sarimax' else (0,0,0,0) # Example seasonal for SARIMAX

            if model_name.lower() == 'sarimax':
                 model = SARIMAX(ts_df[target_col], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            else: # ARIMA
                 model = ARIMA(ts_df[target_col], order=order)

            results = model.fit(disp=False)
            # Get forecast
            pred_start = ts_df.index[-1] + pd.Timedelta(days=1 if target_freq=='D' else 30) # Adjust based on freq
            pred_end = pred_start + pd.Timedelta(days=periods if target_freq=='D' else periods*30) # Rough estimate
            forecast_result = results.get_prediction(start=pred_start, end=pred_end, dynamic=False)
            forecast_values = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()

            # Combine history and forecast
            forecast_index = pd.date_range(start=pred_start, periods=periods, freq=target_freq)
            forecast_only_df = pd.DataFrame({'forecast': forecast_values}, index=forecast_index)
            forecast_df = pd.concat([ts_df[[target_col]].rename(columns={target_col:'history'}), forecast_only_df], axis=1)

            logger.info(f"{model_name.upper()} forecast generated.")
            # Create Plotly figure manually
            forecast_fig = go.Figure()
            forecast_fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['history'], mode='lines', name='Historical'))
            forecast_fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['forecast'], mode='lines', name='Forecast', line=dict(dash='dash')))
            # Add confidence intervals
            if not conf_int.empty:
                forecast_fig.add_trace(go.Scatter(x=conf_int.index, y=conf_int.iloc[:, 0], mode='lines', line=dict(width=0), name='Lower CI', showlegend=False))
                forecast_fig.add_trace(go.Scatter(x=conf_int.index, y=conf_int.iloc[:, 1], mode='lines', line=dict(width=0), name='Upper CI', fill='tonexty', fillcolor='rgba(255,0,0,0.1)', showlegend=False))
            forecast_fig.update_layout(title=f"{model_name.upper()} Forecast: {target_col}")


        # Add 'Simple Average' or other basic models here...

        else:
            raise ValueError(f"Unsupported forecast model: '{model_name}'. Choose from: prophet, arima, sarimax, simple.")

        return forecast_df, forecast_fig

    except ImportError as e:
         logger.error(f"Import error during forecasting: {e}")
         st.error(f"Missing Library: Please install '{str(e).split(' ')[-1]}' for {model_name} forecasting.") # Try to get lib name
         return None, None
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Data or configuration error during forecasting: {e}", exc_info=True)
        st.error(f"Forecast Error: {e}") # Show config error to user
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error during {model_name} forecast: {e}", exc_info=True)
        st.error(f"Unexpected Forecast Error: {e}")
        return None, None