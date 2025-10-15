import pandas as pd
import numpy as np
import numpy_financial as npf
from scipy import stats
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import statsmodels.api as sm
from scipy.stats import skew, kurtosis, norm
from datetime import datetime, timedelta
import io
import warnings
from typing import Dict, List, Tuple, Optional, Union
warnings.filterwarnings('ignore')


# ─── Configuration and Styling ─────────────────────────────────────────────
st.set_page_config(
    page_title="Merrimac Portfolio Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with Logo Support
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f4e79;
        padding-bottom: 1rem;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 2rem;
    }
    .info-button {
        background-color: #17a2b8;
        color: white;
        border: none;
        padding: 5px 10px;
        border-radius: 50%;
        cursor: pointer;
        font-size: 12px;
        margin-left: 5px;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f4e79;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .quartile-q1 { background-color: #28a745; color: white; padding: 0.2rem 0.4rem; border-radius: 3px; }
    .quartile-q2 { background-color: #20c997; color: white; padding: 0.2rem 0.4rem; border-radius: 3px; }
    .quartile-q3 { background-color: #ffc107; color: black; padding: 0.2rem 0.4rem; border-radius: 3px; }
    .quartile-q4 { background-color: #dc3545; color: white; padding: 0.2rem 0.4rem; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── Taleb Risk Analysis Module ─────────────────────────────────────────────
class TalebRiskAnalyzer:
    """Encapsulates all Taleb-inspired risk calculations"""
    
    def __init__(self, returns_data: pd.DataFrame):
        self.returns_data = returns_data
        self.fund_columns = [col for col in returns_data.columns if col != 'Date']
    
    def calculate_tail_metrics(self, fund_returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive tail risk metrics for a fund"""
        
        if len(fund_returns) < 24:
            return {}
        
        sorted_returns = np.sort(fund_returns.dropna())
        n = len(sorted_returns)
        
        metrics = {
            'n_observations': n,
            'mean': fund_returns.mean(),
            'std': fund_returns.std(),
            'skewness': skew(fund_returns),
            'excess_kurtosis': kurtosis(fund_returns),
            'min': sorted_returns[0],
            'p1': sorted_returns[int(0.01 * n)] if n > 100 else sorted_returns[0],
            'p5': sorted_returns[int(0.05 * n)] if n > 20 else sorted_returns[1],
            'p95': sorted_returns[int(0.95 * n)] if n > 20 else sorted_returns[-2],
            'p99': sorted_returns[int(0.99 * n)] if n > 100 else sorted_returns[-1],
            'max': sorted_returns[-1]
        }
        
        # Tail ratios (Taleb's approach)
        metrics['left_tail_ratio'] = abs(metrics['p1'] / metrics['p5']) if metrics['p5'] != 0 else np.nan
        metrics['right_tail_ratio'] = metrics['p99'] / metrics['p95'] if metrics['p95'] != 0 else np.nan
        
        # Hill estimator for tail index
        metrics['tail_index'] = self._calculate_hill_estimator(sorted_returns)
        
        return metrics
    
    def _calculate_hill_estimator(self, sorted_returns: np.ndarray, tail_fraction: float = 0.1) -> float:
        """Calculate Hill estimator for power law tail index"""
        k = int(tail_fraction * len(sorted_returns))
        if k <= 10:
            return np.nan
            
        tail_returns = sorted_returns[:k]
        if tail_returns[0] >= 0:  # No losses in tail
            return np.nan
            
        log_ratios = [np.log(abs(tail_returns[0]) / abs(r)) 
                      for r in tail_returns[1:] if r < 0]
        
        if not log_ratios:
            return np.nan
            
        return len(log_ratios) / sum(log_ratios)
    
    def classify_tail_risk(self, metrics: Dict[str, float]) -> Tuple[str, str]:
        """Classify tail risk level based on metrics"""
        excess_kurt = metrics.get('excess_kurtosis', 0)
        left_tail = metrics.get('left_tail_ratio', 1)
        
        if excess_kurt > 5 and left_tail > 2:
            return "EXTREME", "red"
        elif excess_kurt > 3 or left_tail > 1.5:
            return "HIGH", "orange"
        elif excess_kurt > 1:
            return "MODERATE", "yellow"
        else:
            return "LOW", "green"
    
    def run_fat_tail_monte_carlo(self, fund_returns: pd.Series, 
                                n_simulations: int = 10000,
                                horizon_months: int = 12) -> Dict:
        """Run Monte Carlo simulation with fat-tailed distributions"""
        
        if len(fund_returns) < 24:
            return {}
        
        # Fit Student's t-distribution
        params = stats.t.fit(fund_returns.dropna())
        df, loc, scale = params
        
        # Vectorized simulation for performance
        if df < 30:  # Use t-distribution for fat tails
            all_returns = stats.t.rvs(df, loc=loc, scale=scale, 
                                     size=(n_simulations, horizon_months))
        else:  # Fall back to empirical resampling
            all_returns = np.random.choice(fund_returns.dropna(), 
                                         size=(n_simulations, horizon_months), 
                                         replace=True)
        
        # Vectorized terminal value calculation
        terminal_values = np.prod(1 + all_returns, axis=1)
        
        # Sample paths for visualization (first 100)
        sample_paths = [np.cumprod(1 + all_returns[i]) for i in range(min(100, n_simulations))]
        
        return {
            'terminal_values': terminal_values,
            'mean_return': np.mean(terminal_values) - 1,
            'median_return': np.median(terminal_values) - 1,
            'sample_paths': sample_paths,
            'tail_parameter': df,
            'var_95': np.percentile(terminal_values - 1, 5),
            'cvar_95': np.mean(terminal_values[terminal_values <= np.percentile(terminal_values, 5)] - 1),
            'var_99': np.percentile(terminal_values - 1, 1),
            'cvar_99': np.mean(terminal_values[terminal_values <= np.percentile(terminal_values, 1)] - 1)
        }
    
    def detect_nonlinearity(self, fund_returns: pd.Series, 
                           market_returns: pd.Series) -> Dict:
        """Detect nonlinearities and convexity in fund returns"""
        
        # Align data
        aligned_data = pd.concat([fund_returns, market_returns], axis=1, join='inner').dropna()
        
        if len(aligned_data) < 24:
            return {}
        
        fund_ret = aligned_data.iloc[:, 0]
        market_ret = aligned_data.iloc[:, 1]
        
        # Linear regression
        X_linear = sm.add_constant(market_ret)
        linear_model = sm.OLS(fund_ret, X_linear).fit()
        
        # Nonlinear regression (quadratic)
        market_squared = market_ret ** 2
        X_nonlinear = sm.add_constant(pd.DataFrame({
            'market': market_ret,
            'market_squared': market_squared
        }))
        nonlinear_model = sm.OLS(fund_ret, X_nonlinear).fit()
        
        # Separate up/down market analysis
        up_market = market_ret > 0
        down_market = market_ret < 0
        
        up_beta = None
        down_beta = None
        
        if up_market.sum() > 10:
            X_up = sm.add_constant(market_ret[up_market])
            up_model = sm.OLS(fund_ret[up_market], X_up).fit()
            up_beta = up_model.params[1]
        
        if down_market.sum() > 10:
            X_down = sm.add_constant(market_ret[down_market])
            down_model = sm.OLS(fund_ret[down_market], X_down).fit()
            down_beta = down_model.params[1]
        
        return {
            'linear_beta': linear_model.params[1],
            'linear_r2': linear_model.rsquared,
            'convexity': nonlinear_model.params.get('market_squared', 0),
            'nonlinear_r2': nonlinear_model.rsquared,
            'up_beta': up_beta,
            'down_beta': down_beta,
            'has_negative_convexity': nonlinear_model.params.get('market_squared', 0) < -0.5
        }
    
    def analyze_hidden_risks(self, fund_returns: pd.Series, 
                           other_fund_returns: List[pd.Series]) -> Dict:
        """Analyze hidden risks including correlation instability and clustering"""
        
        if len(fund_returns) < 36:
            return {}
        
        risks = {}
        
        # 1. Serial correlation
        risks['serial_correlation'] = fund_returns.autocorr(lag=1)
        
        # 2. Correlation instability
        correlations = []
        for other_returns in other_fund_returns:
            aligned = pd.concat([fund_returns, other_returns], axis=1, join='inner')
            if len(aligned) > 24:
                rolling_corr = aligned.iloc[:, 0].rolling(12).corr(aligned.iloc[:, 1])
                correlations.extend(rolling_corr.dropna().tolist())
        
        if correlations:
            risks['corr_instability'] = np.std(correlations)
            risks['max_correlation'] = np.max(correlations)
        
        # 3. Drawdown clustering
        bad_months = fund_returns < fund_returns.quantile(0.1)
        runs = self._calculate_runs(bad_months)
        
        if runs:
            risks['max_bad_streak'] = max(runs)
            risks['avg_bad_streak'] = np.mean(runs)
        
        # 4. Liquidity proxy - FIXED SECTION
        if len(fund_returns) > 2:
            # Calculate return reversals manually
            returns_array = fund_returns.values
            reversals = []
            for i in range(1, len(returns_array)):
                reversal = -returns_array[i-1] * returns_array[i]
                reversals.append(reversal)
            
            risks['illiquidity_score'] = np.mean(reversals) if reversals else 0
        
        return risks
   
    
    def _calculate_runs(self, binary_series: pd.Series) -> List[int]:
        """Calculate lengths of consecutive True values"""
        runs = []
        current_run = 0
        
        for value in binary_series:
            if value:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        
        if current_run > 0:
            runs.append(current_run)
            
        return runs
    
    def calculate_risk_score(self, hidden_risks: Dict) -> Tuple[int, List[str]]:
        """Calculate overall risk score and identify red flags"""
        score = 0
        flags = []
        
        # Serial correlation risk
        if 'serial_correlation' in hidden_risks and abs(hidden_risks['serial_correlation']) > 0.3:
            score += 1
            flags.append("High Serial Correlation")
        
        # Correlation instability risk
        if 'corr_instability' in hidden_risks and hidden_risks['corr_instability'] > 0.3:
            score += 2
            flags.append("Unstable Correlations")
        
        # Clustering risk
        if 'max_bad_streak' in hidden_risks and hidden_risks['max_bad_streak'] > 3:
            score += 2
            flags.append("Loss Clustering")
        
        # Illiquidity risk
        if 'illiquidity_score' in hidden_risks and hidden_risks['illiquidity_score'] > 0.1:
            score += 1
            flags.append("Liquidity Concerns")
        
        return score, flags

# ─── Helper Functions ─────────────────────────────────────────────
def show_info_button(key: str, title: str, content: str):
    """Display an info button with expandable content"""
    if st.button(f"ℹ️ {title}", key=key, help="Click for more information"):
        st.info(content)

@st.cache_data
def calculate_drawdown_series(returns: pd.Series) -> pd.Series:
    """Calculate drawdown series from returns"""
    try:
        returns_clean = returns.dropna()
        if len(returns_clean) == 0:
            return pd.Series()
        
        cum_returns = (1 + returns_clean).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown
    except Exception as e:
        st.warning(f"Error calculating drawdown: {e}")
        return pd.Series()

@st.cache_data
def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from returns series"""
    try:
        drawdown = calculate_drawdown_series(returns)
        return drawdown.min() if len(drawdown) > 0 else np.nan
    except:
        return np.nan

@st.cache_data
def downside_risk(returns: pd.Series, target_return: float = 0) -> float:
    """Calculate downside deviation"""
    try:
        returns_clean = returns.dropna()
        if len(returns_clean) < 2:
            return np.nan
            
        downside_returns = returns_clean[returns_clean < target_return] - target_return
        if len(downside_returns) == 0:
            return 0.0
        return np.sqrt((downside_returns**2).mean()) * np.sqrt(12)
    except Exception:
        return np.nan

@st.cache_data
def calculate_var_cvar(returns: pd.Series, confidence=0.95):
    """Calculate Value at Risk and Conditional Value at Risk"""
    try:
        returns_clean = returns.dropna()
        if len(returns_clean) < 2:
            return np.nan, np.nan
        
        var = np.percentile(returns_clean, (1 - confidence) * 100)
        cvar = returns_clean[returns_clean <= var].mean()
        return var, cvar
    except Exception:
        return np.nan, np.nan

@st.cache_data
def sortino_ratio(returns: pd.Series, target_return: float = 0) -> float:
    """Calculate Sortino ratio"""
    try:
        returns_clean = returns.dropna()
        if len(returns_clean) < 2:
            return np.nan
            
        excess_return = returns_clean.mean() - target_return/12
        downside_returns = returns_clean[returns_clean < target_return/12] - target_return/12
        if len(downside_returns) == 0:
            return np.nan
            
        downside_std = np.sqrt((downside_returns ** 2).mean())
        return excess_return / downside_std * np.sqrt(12) if downside_std != 0 else np.nan
    except Exception:
        return np.nan

@st.cache_data
def omega_ratio(returns: pd.Series, threshold: float = 0) -> float:
    """Calculate Omega ratio"""
    try:
        returns_clean = returns.dropna()
        if len(returns_clean) < 2:
            return np.nan
            
        threshold_monthly = threshold / 12
        gains = returns_clean[returns_clean > threshold_monthly] - threshold_monthly
        losses = threshold_monthly - returns_clean[returns_clean < threshold_monthly]
        
        if gains.sum() == 0 or losses.sum() == 0:
            return np.nan
            
        return gains.sum() / losses.sum()
    except Exception:
        return np.nan

@st.cache_data
def compute_metrics(returns: pd.Series, benchmark: pd.Series = None) -> dict:
    """Compute comprehensive performance metrics"""
    try:
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            return {k: np.nan for k in ['Ann_Return', 'Ann_Vol', 'Sharpe', 'Max_DD']}
        
        if benchmark is not None:
            benchmark_clean = benchmark.dropna()
            # Align the series
            combined = pd.concat([returns_clean, benchmark_clean], axis=1, join='inner').dropna()
            if len(combined) == 0:
                benchmark = None
            else:
                returns_clean = combined.iloc[:, 0]
                benchmark = combined.iloc[:, 1]
        
        # Basic metrics
        if len(returns_clean) > 0:
            ann_return = (1 + returns_clean).prod()**(12/len(returns_clean)) - 1
        else:
            ann_return = np.nan
            
        ann_vol = returns_clean.std() * np.sqrt(12) if len(returns_clean) > 1 else np.nan
        sharpe = returns_clean.mean() / returns_clean.std() * np.sqrt(12) if (len(returns_clean) > 1 and returns_clean.std() != 0) else np.nan
        
        dd_series = calculate_drawdown_series(returns_clean)
        max_dd = dd_series.min() if len(dd_series) > 0 else np.nan
        
        # Enhanced risk metrics
        var_95, cvar_95 = calculate_var_cvar(returns_clean, 0.95)
        sortino = sortino_ratio(returns_clean)
        omega = omega_ratio(returns_clean)
        
        metrics = {
            'Ann_Return': ann_return,
            'Ann_Vol': ann_vol,
            'Sharpe': sharpe,
            'Sortino': sortino,
            'Omega': omega,
            'Max_DD': max_dd,
            'Calmar': ann_return / abs(max_dd) if (max_dd < 0 and not np.isnan(max_dd)) else np.nan,
            'Skew': skew(returns_clean) if len(returns_clean) > 2 else np.nan,
            'Kurtosis': kurtosis(returns_clean) if len(returns_clean) > 3 else np.nan,
            'Downside_Risk': downside_risk(returns_clean),
            'VaR_95': var_95,
            'CVaR_95': cvar_95,
            'Pct_Positive_Months': (returns_clean > 0).mean() * 100 if len(returns_clean) > 0 else np.nan
        }
        
        # Benchmark-relative metrics
        if benchmark is not None:
            try:
                X = sm.add_constant(benchmark)
                reg = sm.OLS(returns_clean, X).fit()
                bench_metrics = compute_metrics(benchmark)
                te = (returns_clean - benchmark).std() * np.sqrt(12)
                
                metrics.update({
                    'Alpha': reg.params[0] * 12,  # Annualized
                    'Beta': reg.params[1] if len(reg.params) > 1 else np.nan,
                    'R_Squared': reg.rsquared,
                    'Correlation': returns_clean.corr(benchmark),
                    'Tracking_Error': te,
                    'Info_Ratio': (ann_return - bench_metrics['Ann_Return']) / te if te != 0 else np.nan,
                    'Treynor_Ratio': (ann_return - 0.02) / reg.params[1] if (len(reg.params) > 1 and reg.params[1] != 0) else np.nan
                })
            except Exception as e:
                metrics.update({
                    'Alpha': np.nan, 'Beta': np.nan, 'R_Squared': np.nan,
                    'Correlation': np.nan, 'Tracking_Error': np.nan, 'Info_Ratio': np.nan,
                    'Treynor_Ratio': np.nan
                })
        
        return metrics
        
    except Exception as e:
        st.error(f"Error computing metrics: {e}")
        return {k: np.nan for k in ['Ann_Return', 'Ann_Vol', 'Sharpe', 'Max_DD']}

def validate_data(data, data_type="returns"):
    """Validate data quality and format"""
    issues = []
    
    if data is None or data.empty:
        issues.append(f"{data_type} data is empty")
        return issues
    
    if data_type == "returns":
        # Check for date column
        if 'Date' not in data.columns:
            issues.append("No 'Date' column found in returns data")
        else:
            # Try to convert to datetime
            try:
                pd.to_datetime(data['Date'])
            except:
                issues.append("Date column cannot be converted to datetime")
        
        # Check for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            issues.append("No numeric return columns found")
        
        # Check for excessive missing values
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        missing_percentage = missing_cells / total_cells
        
        if missing_percentage > 0.8:  # Only flag if more than 80% missing
            issues.append(f"Extremely high percentage of missing values: {missing_percentage:.1%}")
        elif missing_percentage > 0.5:  # Warn but don't block if 50-80% missing
            st.warning(f"⚠️ High percentage of missing values: {missing_percentage:.1%} - This is common with hedge fund data where funds have different start dates")
    
    return issues

def export_data_to_excel(data_dict: dict, filename: str = "portfolio_analysis.xlsx"):
    """Export analysis results to Excel with multiple sheets"""
    buffer = io.BytesIO()
    
    try:
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            for sheet_name, data in data_dict.items():
                if isinstance(data, pd.DataFrame):
                    data.to_excel(writer, sheet_name=sheet_name, index=True)
                else:
                    pd.DataFrame([data]).to_excel(writer, sheet_name=sheet_name, index=False)
        
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error exporting to Excel: {e}")
        return None










# ─── Private Equity Specific Functions ─────────────────────────────────────
@st.cache_data
def calculate_irr(cash_flows: list, guess: float = 0.1) -> float:
    """Calculate IRR from cash flows"""
    try:
        if len(cash_flows) < 2:
            return np.nan
        return npf.irr(cash_flows)
    except Exception:
        return np.nan

@st.cache_data  
def calculate_moic(invested: float, distributions: float, nav: float) -> float:
    """Calculate Multiple of Invested Capital"""
    try:
        if invested <= 0:
            return np.nan
        return (distributions + nav) / invested
    except Exception:
        return np.nan

@st.cache_data
def calculate_dpi(invested: float, distributions: float) -> float:
    """Calculate Distributions to Paid-In Capital"""
    try:
        if invested <= 0:
            return np.nan
        return distributions / invested
    except Exception:
        return np.nan

@st.cache_data
def calculate_rvpi(invested: float, nav: float) -> float:
    """Calculate Residual Value to Paid-In Capital"""
    try:
        if invested <= 0:
            return np.nan
        return nav / invested
    except Exception:
        return np.nan

@st.cache_data
def calculate_tvpi(invested: float, distributions: float, nav: float) -> float:
    """Calculate Total Value to Paid-In Capital"""
    try:
        if invested <= 0:
            return np.nan
        return (distributions + nav) / invested
    except Exception:
        return np.nan

def calculate_pe_metrics(df_row) -> dict:
    """Calculate comprehensive PE metrics for a fund using new format"""
    try:
        metrics = {}
        
        # Direct mappings from new format
        metrics['MOIC'] = df_row.get('Net_MOIC', np.nan)
        metrics['Gross_MOIC'] = df_row.get('Gross_MOIC', np.nan)
        metrics['DPI'] = df_row.get('Net_DPI', np.nan)
        metrics['IRR'] = df_row.get('Net_IRR', np.nan)
        
        # Calculate RVPI (Residual Value to Paid-In)
        # RVPI = (Latest Valuation - Distributions) / Paid-in Capital
        # Since MOIC = (Distributions + NAV) / Paid-in Capital
        # and DPI = Distributions / Paid-in Capital
        # then RVPI = MOIC - DPI
        
        paid_in = df_row.get('Paid_in_Capital', 0)
        latest_val = df_row.get('Latest_Valuation', 0)
        
        if paid_in > 0:
            # Calculate implied distributions from DPI
            distributions = metrics['DPI'] * paid_in
            # RVPI is the remaining value divided by paid-in capital
            metrics['RVPI'] = (latest_val / paid_in) - metrics['DPI']
            metrics['TVPI'] = metrics['DPI'] + metrics['RVPI']  # Total Value to Paid-In
            
            # For compatibility with existing code
            metrics['Percent_Called'] = 100.0  # Assume fully called if we have paid-in capital
        else:
            metrics['RVPI'] = np.nan
            metrics['TVPI'] = np.nan
            metrics['Percent_Called'] = 0
        
        # Add quartile ranking
        metrics['DPI_Quartile'] = df_row.get('Net_DPI_Quartile_Ranking', np.nan)
            
        return metrics
        
    except Exception as e:
        return {'MOIC': np.nan, 'DPI': np.nan, 'RVPI': np.nan, 'TVPI': np.nan, 'IRR': np.nan, 'Percent_Called': np.nan}

def generate_j_curve_data(vintage_year: int, fund_life: int = 10) -> pd.DataFrame:
    """Generate synthetic J-curve data for visualization"""
    try:
        # Typical J-curve pattern
        years = list(range(vintage_year, vintage_year + fund_life + 1))
        
        # Synthetic IRR progression (typical J-curve)
        irr_progression = []
        for i in range(len(years)):
            if i == 0:
                irr = 0  # Start
            elif i <= 2:
                irr = -10 - (i * 5)  # Initial negative returns
            elif i <= 4:
                irr = -15 + (i - 2) * 8  # Recovery
            elif i <= 6:
                irr = 1 + (i - 4) * 6  # Growth phase
            else:
                irr = 13 + (i - 6) * 2  # Maturation
            
            irr_progression.append(max(-25, min(irr, 25)))  # Cap between -25% and 25%
        
        return pd.DataFrame({
            'Year': years,
            'IRR': irr_progression
        })
        
    except Exception:
        # Return empty DataFrame if generation fails
        return pd.DataFrame({'Year': [], 'IRR': []})

@st.cache_data
def calculate_fund_metrics_batch(returns_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics for all funds in batch for better performance"""
    fund_columns = [col for col in returns_data.columns if col != 'Date']
    metrics_dict = {}
    
    for fund in fund_columns:
        try:
            fund_returns = returns_data[fund].dropna()
            if len(fund_returns) > 0:
                metrics_dict[fund] = compute_metrics(fund_returns)
        except Exception as e:
            st.warning(f"Error calculating metrics for {fund}: {e}")
            continue
    
    if metrics_dict:
        return pd.DataFrame(metrics_dict).T
    else:
        return pd.DataFrame()

def analyze_recovery(returns: pd.Series, worst_date: pd.Timestamp, window_size: int) -> Dict:
    """Analyze recovery from worst period"""
    try:
        # Find the index of worst date
        date_index = returns.index.get_loc(worst_date)
        
        # Get value at start of worst period
        start_index = max(0, date_index - window_size + 1)
        start_value = (1 + returns.iloc[:start_index]).prod()
        trough_value = (1 + returns.iloc[:date_index+1]).prod()
        
        # Track recovery
        recovery_path = []
        recovery_months = -1
        
        for i in range(date_index + 1, len(returns)):
            current_value = (1 + returns.iloc[:i+1]).prod()
            recovery_path.append(current_value / trough_value)
            
            if current_value >= start_value and recovery_months == -1:
                recovery_months = i - date_index
        
        return {
            'recovery_months': recovery_months,
            'recovery_path': recovery_path if recovery_path else None,
            'permanent_loss': recovery_months == -1
        }
    except:
        return {
            'recovery_months': -1,
            'recovery_path': None,
            'permanent_loss': True
        }

@st.cache_data
def calculate_recovery_time(drawdown_series: pd.Series) -> float:
    """Calculate average recovery time from drawdowns"""
    try:
        if len(drawdown_series) == 0:
            return np.nan
        
        recovery_times = []
        in_drawdown = False
        drawdown_start = 0
        
        for i, dd in enumerate(drawdown_series):
            if dd < -0.01 and not in_drawdown:  # Start of drawdown
                in_drawdown = True
                drawdown_start = i
            elif dd >= -0.01 and in_drawdown:  # Recovery
                recovery_time = i - drawdown_start
                recovery_times.append(recovery_time)
                in_drawdown = False
        
        return np.mean(recovery_times) if recovery_times else np.nan
        
    except Exception:
        return np.nan

@st.cache_data
def calculate_factor_metrics(fund_returns: pd.Series, benchmark_returns: pd.Series, fund_name: str) -> dict:
    """Calculate comprehensive factor metrics including Bull/Bear Beta"""
    
    try:
        # Basic regression
        X = sm.add_constant(benchmark_returns)
        model = sm.OLS(fund_returns, X).fit()
        
        # Basic metrics
        alpha = model.params[0] * 12  # Annualized
        beta = model.params[1]
        r_squared = model.rsquared
        alpha_pvalue = model.pvalues[0]
        beta_pvalue = model.pvalues[1]
        
        # Bull and Bear Beta
        bull_periods = benchmark_returns > 0
        bear_periods = benchmark_returns < 0
        
        bull_beta = np.nan
        bear_beta = np.nan
        bull_alpha = np.nan
        bear_alpha = np.nan
        
        if bull_periods.sum() > 10:  # Need sufficient bull market data
            X_bull = sm.add_constant(benchmark_returns[bull_periods])
            model_bull = sm.OLS(fund_returns[bull_periods], X_bull).fit()
            bull_beta = model_bull.params[1]
            bull_alpha = model_bull.params[0] * 12
        
        if bear_periods.sum() > 10:  # Need sufficient bear market data
            X_bear = sm.add_constant(benchmark_returns[bear_periods])
            model_bear = sm.OLS(fund_returns[bear_periods], X_bear).fit()
            bear_beta = model_bear.params[1]
            bear_alpha = model_bear.params[0] * 12
        
        return {
            'Fund': fund_name,
            'Alpha': alpha,
            'Alpha_PValue': alpha_pvalue,
            'Beta': beta,
            'Beta_PValue': beta_pvalue,
            'Bull_Beta': bull_beta,
            'Bear_Beta': bear_beta,
            'R_Squared': r_squared,
            'Observations': len(fund_returns)
        }
        
    except Exception as e:
        return {
            'Fund': fund_name,
            'Alpha': np.nan,
            'Alpha_PValue': np.nan,
            'Beta': np.nan,
            'Beta_PValue': np.nan,
            'Bull_Beta': np.nan,
            'Bear_Beta': np.nan,
            'R_Squared': np.nan,
            'Observations': 0
        }

@st.cache_data
def calculate_rolling_factor_metrics(fund_returns: pd.Series, benchmark_returns: pd.Series, 
                                   fund_name: str, windows: List[int] = [12, 36]) -> Dict:
    """Calculate rolling alpha and beta for multiple windows"""
    
    rolling_results = {}
    
    for window in windows:
        if len(fund_returns) < window:
            continue
            
        rolling_alpha = []
        rolling_beta = []
        rolling_dates = []
        
        for i in range(window, len(fund_returns) + 1):
            try:
                fund_window = fund_returns.iloc[i-window:i]
                bench_window = benchmark_returns.iloc[i-window:i]
                
                X = sm.add_constant(bench_window)
                model = sm.OLS(fund_window, X).fit()
                
                rolling_alpha.append(model.params[0] * 12)  # Annualized
                rolling_beta.append(model.params[1])
                rolling_dates.append(fund_returns.index[i-1])
                
            except:
                rolling_alpha.append(np.nan)
                rolling_beta.append(np.nan)
                rolling_dates.append(fund_returns.index[i-1])
        
        rolling_results[f'{window}M'] = {
            'dates': rolling_dates,
            'alpha': rolling_alpha,
            'beta': rolling_beta
        }
    
    return rolling_results

# ─── Complete Private Equity Analysis ──────────────────────────────────────

# Add this function before show_private_equity_page()

def get_default_pe_data():
    """Get default PE data in the new format"""
    return {
        "Fund_Name": ["Apollo Fund VIII", "Blackstone Capital VII", "KKR Americas XII", 
                      "Carlyle Partners VI", "TPG Partners VII", "Bain Capital XI", 
                      "Vista Equity V", "Leonard Green IV", "Advent GPE VIII", "Warburg Pincus XII"],
        "Paid_in_Capital": [50, 75, 40, 30, 60, 35, 25, 20, 45, 30],
        "Latest_Valuation": [85, 103, 55, 42, 85, 53, 41, 28, 65, 42],
        "Valuation_Date": ["2024-12-31", "2024-12-31", "2024-12-31", "2024-12-31", "2024-12-31",
                          "2024-12-31", "2024-12-31", "2024-12-31", "2024-12-31", "2024-12-31"],
        "Gross_MOIC": [1.85, 1.65, 2.10, 1.55, 1.75, 1.80, 2.20, 2.05, 1.60, 1.70],
        "Net_MOIC": [1.70, 1.47, 1.88, 1.40, 1.58, 1.62, 1.98, 1.85, 1.44, 1.53],
        "Net_DPI": [0.85, 0.75, 0.95, 0.80, 0.70, 0.60, 0.65, 0.90, 0.70, 0.75],
        "Net_IRR": [18.5, 15.2, 21.3, 12.7, 19.0, 16.8, 24.2, 22.1, 14.8, 15.5],
        "Net_DPI_Quartile_Ranking": [1, 2, 1, 2, 3, 3, 2, 1, 3, 2]
    }


def show_pe_performance_analysis(pe_data):
    """New performance analysis tab for detailed metrics"""
    st.subheader("📈 Performance Analysis")
    
    # IRR vs MOIC scatter
    if all(col in pe_data.columns for col in ['Net_IRR', 'Net_MOIC']):
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter = px.scatter(
                pe_data,
                x='Net_IRR',
                y='Net_MOIC',
                size='Paid_in_Capital',
                color='Net_DPI_Quartile_Ranking',
                hover_name='Fund_Name',
                title="IRR vs MOIC Analysis",
                color_continuous_scale='RdYlGn_r',
                labels={'Net_DPI_Quartile_Ranking': 'DPI Quartile'}
            )
            
            fig_scatter.update_layout(
                xaxis_title="Net IRR (%)",
                yaxis_title="Net MOIC",
                height=500
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # DPI vs Latest Valuation
            if 'Net_DPI' in pe_data.columns:
                # Calculate RVPI proxy
                pe_data['RVPI_proxy'] = pe_data['Net_MOIC'] - pe_data['Net_DPI']
                
                fig_dpi = px.scatter(
                    pe_data,
                    x='Net_DPI',
                    y='RVPI_proxy',
                    size='Paid_in_Capital',
                    hover_name='Fund_Name',
                    title="DPI vs Residual Value Analysis"
                )
                
                fig_dpi.update_layout(
                    xaxis_title="Net DPI (Realized)",
                    yaxis_title="RVPI (Unrealized)",
                    height=500
                )
                
                # Add diagonal line for total value
                max_val = max(pe_data['Net_DPI'].max(), pe_data['RVPI_proxy'].max())
                fig_dpi.add_trace(go.Scatter(
                    x=[0, max_val],
                    y=[max_val, 0],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    name='TVPI = 1.0x',
                    showlegend=False
                ))
                
                st.plotly_chart(fig_dpi, use_container_width=True)
    
    # Performance metrics by quartile
    st.subheader("📊 Performance by Quartile")
    
    if 'Net_DPI_Quartile_Ranking' in pe_data.columns:
        # Create box plots for each metric by quartile
        metrics_to_plot = ['Net_IRR', 'Net_MOIC', 'Gross_MOIC', 'Net_DPI']
        available_metrics = [m for m in metrics_to_plot if m in pe_data.columns]
        
        if available_metrics:
            fig_box = make_subplots(
                rows=2, cols=2,
                subplot_titles=available_metrics
            )
            
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            
            for i, metric in enumerate(available_metrics[:4]):
                row, col = positions[i]
                
                for quartile in sorted(pe_data['Net_DPI_Quartile_Ranking'].unique()):
                    quartile_data = pe_data[pe_data['Net_DPI_Quartile_Ranking'] == quartile]
                    
                    fig_box.add_trace(
                        go.Box(
                            y=quartile_data[metric],
                            name=f'Q{quartile}',
                            boxmean='sd',
                            marker_color=['#28a745', '#20c997', '#ffc107', '#dc3545'][quartile-1]
                        ),
                        row=row, col=col
                    )
            
            fig_box.update_layout(
                height=800,
                showlegend=False,
                title_text="Performance Metrics Distribution by Quartile"
            )
            
            st.plotly_chart(fig_box, use_container_width=True)



def show_complete_private_equity_page():
    """Complete Private Equity Fund Analysis - main function update"""
    st.markdown('<h1 class="main-header">🏢 Private Equity Fund Analysis</h1>', unsafe_allow_html=True)
    
    # Use the new default data structure
    default_pe_data = get_default_pe_data()
    
    # Load data
    pe_data = load_pe_data(default_pe_data)
    
    if pe_data is not None and not pe_data.empty:
        # Apply filters
        filtered_pe_data = apply_pe_filters(pe_data)
        
        if not filtered_pe_data.empty:
            # Calculate enhanced metrics
            enhanced_pe_data = calculate_enhanced_pe_metrics(filtered_pe_data)
            
            # Main tabs - keep most existing tabs but update content
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Portfolio Overview",
                "🏆 Performance Rankings", 
                "💰 Cash Flow Analytics",
                "📈 Performance Analysis",
                "🔍 Due Diligence",
                "📋 Reports & Export"
            ])
            
            with tab1:
                show_pe_portfolio_overview(enhanced_pe_data)
            
            with tab2:
                show_pe_performance_rankings(enhanced_pe_data)
            
            with tab3:
                show_pe_cashflow_analytics(enhanced_pe_data)
            
            with tab4:
                # New performance analysis tab
                show_pe_performance_analysis(enhanced_pe_data)
            
            with tab5:
                show_pe_due_diligence(enhanced_pe_data)
            
            with tab6:
                show_pe_reports_export(enhanced_pe_data)
        
        else:
            st.warning("⚠️ No funds match your current filters. Please adjust the criteria.")
    
    else:
        st.info("👆 Please upload PE data or use the default dataset to begin analysis.")

    return pe_data


def load_pe_data(default_data):
    """Load PE data from file upload or use default"""
    
    st.subheader("📁 Private Equity Data Loading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "📊 Upload PE Excel File",
            type=['xlsx', 'csv'],
            help="Upload file with PE fund data",
            key="pe_file_upload"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_columns = [
                    'Fund_Name', 'Paid_in_Capital', 'Latest_Valuation', 
                    'Valuation_Date', 'Gross_MOIC', 'Net_MOIC', 
                    'Net_DPI', 'Net_IRR', 'Net_DPI_Quartile_Ranking'
                ]
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                    st.info("Using default PE dataset for demonstration")
                    return pd.DataFrame(default_data)
                
                # Convert date column
                df['Valuation_Date'] = pd.to_datetime(df['Valuation_Date'])
                
                st.success(f"✅ Data loaded: {len(df)} funds")
                return df
                
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
                return pd.DataFrame(default_data)
        else:
            st.info("Using default PE dataset for demonstration")
            return pd.DataFrame(default_data)
    
    with col2:
        st.subheader("📥 Download Template")
        if st.button("📄 Get PE Template", key="pe_template_btn"):
            # Create template with new format
            template_data = {
                "Fund_Name": ["Example Fund I", "Example Fund II"],
                "Paid_in_Capital": [50.0, 75.0],
                "Latest_Valuation": [85.0, 110.0],
                "Valuation_Date": ["2024-12-31", "2024-12-31"],
                "Gross_MOIC": [1.85, 1.65],
                "Net_MOIC": [1.70, 1.47],
                "Net_DPI": [0.85, 0.75],
                "Net_IRR": [18.5, 15.2],
                "Net_DPI_Quartile_Ranking": [1, 2]
            }
            
            template_df = pd.DataFrame(template_data)
            buffer = export_data_to_excel({"PE_Template": template_df}, "PE_Template.xlsx")
            if buffer:
                st.download_button(
                    label="📥 Download Excel Template",
                    data=buffer,
                    file_name="PE_Template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="pe_template_download"
                )



def apply_pe_filters(pe_data):
    """Apply comprehensive PE filters - updated for new format"""
    
    with st.sidebar:
        st.header("🎛️ Private Equity Filters")
        
        # Basic stats
        st.metric("Total PE Funds", len(pe_data))
        
        filtered_data = pe_data.copy()
        
        # 1. Valuation Date Filter (replacing Vintage Year)
        if 'Valuation_Date' in pe_data.columns:
            try:
                # Ensure Valuation_Date is datetime
                filtered_data['Valuation_Date'] = pd.to_datetime(filtered_data['Valuation_Date'], errors='coerce')
                
                # Extract year from valuation date for filtering
                filtered_data['Valuation_Year'] = filtered_data['Valuation_Date'].dt.year
                
                # Check if we have valid years
                valid_years = filtered_data['Valuation_Year'].dropna()
                
                if len(valid_years) > 0:
                    min_year = int(valid_years.min())
                    max_year = int(valid_years.max())
                    
                    year_range = st.slider(
                        "📅 Valuation Year Range",
                        min_value=min_year,
                        max_value=max_year,
                        value=(min_year, max_year),
                        key="pe_valuation_filter"
                    )
                    
                    filtered_data = filtered_data[
                        (filtered_data['Valuation_Year'] >= year_range[0]) & 
                        (filtered_data['Valuation_Year'] <= year_range[1])
                    ]
                else:
                    st.warning("No valid valuation dates found in data")
                    
            except Exception as e:
                st.warning(f"Could not parse valuation dates: {e}")
        
        # 2. Performance Filters
        st.subheader("📊 Performance Criteria")
        
        if 'Net_IRR' in pe_data.columns:
            # Handle potential NaN values in IRR
            irr_values = filtered_data['Net_IRR'].dropna()
            if len(irr_values) > 0:
                min_irr = st.slider(
                    "Minimum Net IRR (%)",
                    min_value=float(irr_values.min()),
                    max_value=float(irr_values.max()),
                    value=float(irr_values.min()),
                    key="pe_min_irr_filter"
                )
                filtered_data = filtered_data[filtered_data['Net_IRR'] >= min_irr]
        
        if 'Net_MOIC' in pe_data.columns:
            # Handle potential NaN values in MOIC
            moic_values = filtered_data['Net_MOIC'].dropna()
            if len(moic_values) > 0:
                min_moic = st.slider(
                    "Minimum Net MOIC",
                    min_value=float(moic_values.min()),
                    max_value=float(moic_values.max()),
                    value=float(moic_values.min()),
                    step=0.1,
                    key="pe_min_moic_filter"
                )
                filtered_data = filtered_data[filtered_data['Net_MOIC'] >= min_moic]
        
        # 3. Quartile Filter
        if 'Net_DPI_Quartile_Ranking' in pe_data.columns:
            # Get unique quartiles, handling NaN
            quartile_values = filtered_data['Net_DPI_Quartile_Ranking'].dropna().unique()
            if len(quartile_values) > 0:
                quartiles = st.multiselect(
                    "🏆 DPI Quartile Ranking",
                    options=sorted([int(q) for q in quartile_values if not pd.isna(q)]),
                    default=sorted([int(q) for q in quartile_values if not pd.isna(q)]),
                    key="pe_quartile_filter"
                )
                
                if quartiles:
                    filtered_data = filtered_data[filtered_data['Net_DPI_Quartile_Ranking'].isin(quartiles)]
        
        # 4. Capital Size Filter
        if 'Paid_in_Capital' in pe_data.columns:
            # Handle potential NaN values in capital
            capital_values = filtered_data['Paid_in_Capital'].dropna()
            if len(capital_values) > 0:
                capital_range = st.slider(
                    "Paid-in Capital Range ($M)",
                    min_value=float(capital_values.min()),
                    max_value=float(capital_values.max()),
                    value=(float(capital_values.min()), float(capital_values.max())),
                    key="pe_capital_filter"
                )
                
                filtered_data = filtered_data[
                    (filtered_data['Paid_in_Capital'] >= capital_range[0]) & 
                    (filtered_data['Paid_in_Capital'] <= capital_range[1])
                ]
        
        # Show results
        st.markdown("---")
        st.metric("Filtered Funds", len(filtered_data))
        
        return filtered_data



def calculate_enhanced_pe_metrics(pe_data):
    """Calculate enhanced PE metrics for all funds"""
    
    enhanced_data = pe_data.copy()
    
    # Calculate core PE metrics for each fund
    for idx, row in enhanced_data.iterrows():
        metrics = calculate_pe_metrics(row)
        for metric, value in metrics.items():
            enhanced_data.loc[idx, metric] = value
    
    # Calculate quartile rankings
    if 'IRR_Pct' in enhanced_data.columns:
        enhanced_data['IRR_Quartile'] = pd.qcut(enhanced_data['IRR_Pct'], 4, labels=['Q4', 'Q3', 'Q2', 'Q1'])
    
    if 'MOIC' in enhanced_data.columns:
        enhanced_data['MOIC_Quartile'] = pd.qcut(enhanced_data['MOIC'], 4, labels=['Q4', 'Q3', 'Q2', 'Q1'])
    
    return enhanced_data

def show_pe_portfolio_overview(pe_data):
    """PE Portfolio Overview - updated for new format"""
    
    st.subheader("📊 Portfolio Overview")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_funds = len(pe_data)
        st.metric("Total Funds", total_funds)
    
    with col2:
        total_capital = pe_data['Paid_in_Capital'].sum() if 'Paid_in_Capital' in pe_data.columns else 0
        st.metric("Total Paid-in Capital", f"${total_capital:.0f}M")
    
    with col3:
        avg_irr = pe_data['Net_IRR'].mean() if 'Net_IRR' in pe_data.columns else 0
        st.metric("Average Net IRR", f"{avg_irr:.1f}%")
    
    with col4:
        avg_moic = pe_data['Net_MOIC'].mean() if 'Net_MOIC' in pe_data.columns else 0
        st.metric("Average Net MOIC", f"{avg_moic:.2f}x")
    
    with col5:
        avg_dpi = pe_data['Net_DPI'].mean() if 'Net_DPI' in pe_data.columns else 0
        st.metric("Average Net DPI", f"{avg_dpi:.2f}x")
    
    # Quartile distribution
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Net_DPI_Quartile_Ranking' in pe_data.columns:
            quartile_counts = pe_data['Net_DPI_Quartile_Ranking'].value_counts().sort_index()
            fig_quartile = px.pie(
                values=quartile_counts.values,
                names=[f"Quartile {q}" for q in quartile_counts.index],
                title="DPI Quartile Distribution",
                color_discrete_map={
                    "Quartile 1": "#28a745",
                    "Quartile 2": "#20c997", 
                    "Quartile 3": "#ffc107",
                    "Quartile 4": "#dc3545"
                }
            )
            st.plotly_chart(fig_quartile, use_container_width=True)
    
    with col2:
        # Gross vs Net MOIC comparison
        if all(col in pe_data.columns for col in ['Gross_MOIC', 'Net_MOIC']):
            fig_moic = go.Figure()
            
            fig_moic.add_trace(go.Bar(
                name='Gross MOIC',
                x=pe_data['Fund_Name'],
                y=pe_data['Gross_MOIC'],
                marker_color='lightblue'
            ))
            
            fig_moic.add_trace(go.Bar(
                name='Net MOIC',
                x=pe_data['Fund_Name'],
                y=pe_data['Net_MOIC'],
                marker_color='darkblue'
            ))
            
            fig_moic.update_layout(
                title="Gross vs Net MOIC Comparison",
                xaxis_tickangle=-45,
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig_moic, use_container_width=True)
    
    # Performance summary table - EDITABLE VERSION
    st.subheader("📋 Fund Performance Summary (Editable)")
    
    display_columns = ['Fund_Name', 'Paid_in_Capital', 'Latest_Valuation', 'Net_IRR', 
                      'Net_MOIC', 'Gross_MOIC', 'Net_DPI', 'Net_DPI_Quartile_Ranking']
    available_columns = [col for col in display_columns if col in pe_data.columns]
    
    if available_columns:
        display_data = pe_data[available_columns].copy()
        
        # Create editable dataframe
        edited_data = st.data_editor(
            display_data,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            disabled=["Fund_Name"],  # Make fund name read-only
            column_config={
                "Paid_in_Capital": st.column_config.NumberColumn(
                    "Paid-in Capital ($M)",
                    min_value=0,
                    step=1.0,
                ),
                "Latest_Valuation": st.column_config.NumberColumn(
                    "Latest Valuation ($M)",
                    min_value=0,
                    step=1.0,
                ),
                "Net_IRR": st.column_config.NumberColumn(
                    "Net IRR (%)",
                    min_value=-100,
                    max_value=200,
                    step=0.1,
                ),
                "Net_MOIC": st.column_config.NumberColumn(
                    "Net MOIC",
                    min_value=0,
                    max_value=10,
                    step=0.01,
                ),
                "Gross_MOIC": st.column_config.NumberColumn(
                    "Gross MOIC",
                    min_value=0,
                    max_value=10,
                    step=0.01,
                ),
                "Net_DPI": st.column_config.NumberColumn(
                    "Net DPI",
                    min_value=0,
                    max_value=10,
                    step=0.01,
                ),
                "Net_DPI_Quartile_Ranking": st.column_config.SelectboxColumn(
                    "DPI Quartile",
                    options=[1, 2, 3, 4],
                    required=True,
                ),
            }
        )
        
        # Add save functionality
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            if st.button("💾 Save Changes", type="primary"):
                # Update the original dataframe
                pe_data.update(edited_data)
                st.success("✅ Changes saved!")
        
        with col2:
            if st.button("📥 Export Edited Data"):
                # Export to Excel
                buffer = export_data_to_excel({"Edited_PE_Data": edited_data}, "Edited_PE_Data.xlsx")
                if buffer:
                    st.download_button(
                        label="Download Excel",
                        data=buffer,
                        file_name=f"Edited_PE_Data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )


def show_pe_performance_rankings(pe_data):
    """PE Performance Rankings - updated for new format"""
    
    st.subheader("🏆 Performance Rankings")
    
    # Scoring weights
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚖️ Scoring Weights")
        
        irr_weight = st.slider("Net IRR Weight", 0.0, 1.0, 0.4, 0.05, key="pe_irr_weight")
        moic_weight = st.slider("Net MOIC Weight", 0.0, 1.0, 0.3, 0.05, key="pe_moic_weight")
        dpi_weight = st.slider("Net DPI Weight", 0.0, 1.0, 0.3, 0.05, key="pe_dpi_weight")
        
        # Normalize weights
        total_weight = irr_weight + moic_weight + dpi_weight
        if total_weight > 0:
            weights = {
                'Net_IRR': irr_weight / total_weight,
                'Net_MOIC': moic_weight / total_weight,
                'Net_DPI': dpi_weight / total_weight
            }
        else:
            weights = {'Net_IRR': 0.4, 'Net_MOIC': 0.3, 'Net_DPI': 0.3}
    
    with col2:
        # Calculate composite scores
        ranking_data = pe_data.copy()
        
        # Normalize each metric (0-1 scale)
        for metric in weights.keys():
            if metric in ranking_data.columns:
                min_val = ranking_data[metric].min()
                max_val = ranking_data[metric].max()
                if max_val > min_val:
                    ranking_data[f'{metric}_Score'] = (ranking_data[metric] - min_val) / (max_val - min_val)
                else:
                    ranking_data[f'{metric}_Score'] = 0.5
        
        # Calculate composite score
        ranking_data['Composite_Score'] = 0
        for metric, weight in weights.items():
            if f'{metric}_Score' in ranking_data.columns:
                ranking_data['Composite_Score'] += ranking_data[f'{metric}_Score'] * weight
        
        # Add rankings
        ranking_data['Rank'] = ranking_data['Composite_Score'].rank(ascending=False, method='min').astype(int)
        
        # Sort by rank
        ranking_data = ranking_data.sort_values('Composite_Score', ascending=False)
        
        # Display rankings
        st.subheader("📊 Fund Rankings")
        
        ranking_columns = ['Rank', 'Fund_Name', 'Net_IRR', 'Net_MOIC', 'Net_DPI', 
                          'Composite_Score', 'Net_DPI_Quartile_Ranking']
        available_ranking_cols = [col for col in ranking_columns if col in ranking_data.columns]
        
        display_ranking = ranking_data[available_ranking_cols].copy()
        
        # Format for display
        if 'Net_IRR' in display_ranking.columns:
            display_ranking['Net_IRR'] = display_ranking['Net_IRR'].apply(lambda x: f"{x:.1f}%")
        
        for col in ['Net_MOIC', 'Net_DPI']:
            if col in display_ranking.columns:
                display_ranking[col] = display_ranking[col].apply(lambda x: f"{x:.2f}x")
        
        if 'Composite_Score' in display_ranking.columns:
            display_ranking['Composite_Score'] = display_ranking['Composite_Score'].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(display_ranking, use_container_width=True, hide_index=True)
    
    # Quartile performance comparison
    st.subheader("📊 Quartile Performance Analysis")
    
    if 'Net_DPI_Quartile_Ranking' in ranking_data.columns:
        quartile_analysis = ranking_data.groupby('Net_DPI_Quartile_Ranking').agg({
            'Net_IRR': ['mean', 'count'],
            'Net_MOIC': 'mean',
            'Net_DPI': 'mean'
        }).round(2)
        
        quartile_analysis.columns = ['Avg_Net_IRR', 'Fund_Count', 'Avg_Net_MOIC', 'Avg_Net_DPI']
        quartile_analysis = quartile_analysis.reset_index()
        
        # Display quartile summary
        col1, col2, col3, col4 = st.columns(4)
        
        quartile_colors = {1: '#28a745', 2: '#20c997', 3: '#ffc107', 4: '#dc3545'}
        
        for i in range(1, 5):
            if i in quartile_analysis['Net_DPI_Quartile_Ranking'].values:
                q_data = quartile_analysis[quartile_analysis['Net_DPI_Quartile_Ranking'] == i].iloc[0]
                
                with [col1, col2, col3, col4][i-1]:
                    st.markdown(f"""
                    <div style="background-color: {quartile_colors[i]}; color: white; padding: 1rem; border-radius: 5px; text-align: center;">
                        <h3>Quartile {i}</h3>
                        <p><strong>{int(q_data['Fund_Count'])} Funds</strong></p>
                        <p>Avg IRR: {q_data['Avg_Net_IRR']:.1f}%</p>
                        <p>Avg MOIC: {q_data['Avg_Net_MOIC']:.2f}x</p>
                        <p>Avg DPI: {q_data['Avg_Net_DPI']:.2f}x</p>
                    </div>
                    """, unsafe_allow_html=True)



def show_pe_vintage_analysis(pe_data):
    """PE Vintage Year Analysis"""
    
    st.subheader("📈 Vintage Year Analysis")
    
    if 'Vintage_Year' in pe_data.columns:
        # Vintage year performance
        vintage_analysis = pe_data.groupby('Vintage_Year').agg({
            'IRR_Pct': ['mean', 'std', 'count'],
            'MOIC': ['mean', 'std'],
            'Commitment_MM': 'sum'
        }).round(2)
        
        vintage_analysis.columns = ['IRR_Mean', 'IRR_Std', 'Fund_Count', 'MOIC_Mean', 'MOIC_Std', 'Total_Commitment']
        vintage_analysis = vintage_analysis.reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # IRR by vintage year with error bars
            fig_vintage_irr = go.Figure()
            
            fig_vintage_irr.add_trace(go.Scatter(
                x=vintage_analysis['Vintage_Year'],
                y=vintage_analysis['IRR_Mean'],
                error_y=dict(type='data', array=vintage_analysis['IRR_Std']),
                mode='markers+lines',
                name='IRR with Std Dev',
                marker=dict(size=vintage_analysis['Fund_Count'] * 3)
            ))
            
            fig_vintage_irr.update_layout(
                title="IRR Performance by Vintage Year",
                xaxis_title="Vintage Year",
                yaxis_title="IRR (%)",
                height=400
            )
            
            st.plotly_chart(fig_vintage_irr, use_container_width=True)
        
        with col2:
            # MOIC by vintage year
            fig_vintage_moic = px.bar(
                vintage_analysis,
                x='Vintage_Year',
                y='MOIC_Mean',
                title="Average MOIC by Vintage Year"
            )
            
            fig_vintage_moic.update_layout(
                yaxis_title="MOIC (x)",
                height=400
            )
            
            st.plotly_chart(fig_vintage_moic, use_container_width=True)
        
        # Vintage year summary table
        st.subheader("📊 Vintage Year Summary")
        
        formatted_vintage = vintage_analysis.copy()
        formatted_vintage['IRR_Mean'] = formatted_vintage['IRR_Mean'].apply(lambda x: f"{x:.1f}%")
        formatted_vintage['IRR_Std'] = formatted_vintage['IRR_Std'].apply(lambda x: f"{x:.1f}%")
        formatted_vintage['MOIC_Mean'] = formatted_vintage['MOIC_Mean'].apply(lambda x: f"{x:.2f}x")
        formatted_vintage['MOIC_Std'] = formatted_vintage['MOIC_Std'].apply(lambda x: f"{x:.2f}x")
        formatted_vintage['Total_Commitment'] = formatted_vintage['Total_Commitment'].apply(lambda x: f"${x:.0f}M")
        
        st.dataframe(formatted_vintage, use_container_width=True, hide_index=True)



def show_pe_cashflow_analytics(pe_data):
    """PE Cash Flow Analytics - updated for new format"""
    
    st.subheader("💰 Cash Flow Analytics")
    
    # Cash flow summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_invested = pe_data['Paid_in_Capital'].sum() if 'Paid_in_Capital' in pe_data.columns else 0
        st.metric("Total Paid-in Capital", f"${total_invested:.0f}M")
    
    with col2:
        # Calculate implied distributions from DPI
        if 'Net_DPI' in pe_data.columns and 'Paid_in_Capital' in pe_data.columns:
            total_distributions = (pe_data['Net_DPI'] * pe_data['Paid_in_Capital']).sum()
            st.metric("Total Distributions", f"${total_distributions:.0f}M")
        else:
            st.metric("Total Distributions", "N/A")
    
    with col3:
        total_valuation = pe_data['Latest_Valuation'].sum() if 'Latest_Valuation' in pe_data.columns else 0
        st.metric("Total Latest Valuation", f"${total_valuation:.0f}M")
    
    # Capital deployment analysis
    if all(col in pe_data.columns for col in ['Paid_in_Capital', 'Latest_Valuation', 'Net_DPI']):
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Value creation waterfall
            fig_waterfall = go.Figure(go.Waterfall(
                name = "20", orientation = "v",
                measure = ["relative", "relative", "total"],
                x = ["Paid-in Capital", "Value Creation", "Latest Valuation"],
                textposition = "outside",
                text = [],
                y = [total_invested, total_valuation - total_invested, total_valuation],
                connector = {"line":{"color":"rgb(63, 63, 63)"}},
            ))
            
            fig_waterfall.update_layout(
                title = "Portfolio Value Creation",
                showlegend = False,
                height=400
            )
            
            st.plotly_chart(fig_waterfall, use_container_width=True)
        
        with col2:
            # MOIC distribution
            if 'Net_MOIC' in pe_data.columns:
                fig_moic_dist = px.histogram(
                    pe_data,
                    x='Net_MOIC',
                    nbins=20,
                    title="Net MOIC Distribution"
                )
                
                fig_moic_dist.update_layout(
                    xaxis_title="Net MOIC",
                    yaxis_title="Number of Funds",
                    height=400
                )
                
                # Add vertical line for breakeven
                fig_moic_dist.add_vline(x=1.0, line_dash="dash", line_color="red", 
                                       annotation_text="Breakeven")
                
                st.plotly_chart(fig_moic_dist, use_container_width=True)



def show_pe_jcurve_analysis(pe_data):
    """PE J-Curve Analysis"""
    
    st.subheader("📉 J-Curve Analysis")
    
    st.info("📊 J-Curve analysis shows typical IRR progression over fund life. Early years typically show negative returns due to fees and deployment lag, followed by value creation in later years.")
    
    # Generate J-curves for selected funds
    if 'Fund_Name' in pe_data.columns and 'Vintage_Year' in pe_data.columns:
        
        selected_funds = st.multiselect(
            "Select Funds for J-Curve Analysis",
            pe_data['Fund_Name'].tolist(),
            default=pe_data['Fund_Name'].tolist()[:3],  # Default first 3 funds
            key="pe_jcurve_funds"
        )
        
        if selected_funds:
            fig_jcurve = go.Figure()
            
            for fund in selected_funds:
                fund_data = pe_data[pe_data['Fund_Name'] == fund].iloc[0]
                vintage_year = int(fund_data['Vintage_Year'])
                
                # Generate J-curve data
                jcurve_data = generate_j_curve_data(vintage_year)
                
                if not jcurve_data.empty:
                    fig_jcurve.add_trace(go.Scatter(
                        x=jcurve_data['Year'],
                        y=jcurve_data['IRR'],
                        mode='lines+markers',
                        name=fund,
                        line=dict(width=2)
                    ))
            
            # Add horizontal line at 0%
            if not jcurve_data.empty:
                fig_jcurve.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="0% IRR")
            
            fig_jcurve.update_layout(
                title="J-Curve Analysis: IRR Progression Over Time",
                xaxis_title="Year",
                yaxis_title="IRR (%)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_jcurve, use_container_width=True)
            
            # J-Curve insights
            st.subheader("💡 J-Curve Insights")
            
            insights = [
                "📉 **Early Years (0-2):** Negative returns due to management fees and investment period",
                "⚖️ **Middle Years (3-5):** Recovery phase as investments mature and first exits occur",
                "📈 **Later Years (6+):** Value creation phase with strong positive returns from exits",
                "🎯 **Peak Performance:** Typically occurs in years 7-10 of fund life"
            ]
            
            for insight in insights:
                st.write(insight)

def show_pe_due_diligence(pe_data):
    """PE Due Diligence Tools"""
    
    st.subheader("🔍 Due Diligence Analysis")
    
    if 'Fund_Name' in pe_data.columns:
        selected_dd_fund = st.selectbox(
            "Select Fund for Due Diligence",
            pe_data['Fund_Name'].tolist(),
            key="pe_dd_fund_select"
        )
        
        if selected_dd_fund:
            fund_data = pe_data[pe_data['Fund_Name'] == selected_dd_fund].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Due Diligence Checklist")
                
                # Define DD criteria
                dd_criteria = {
                    "Track Record (IRR > 15%)": fund_data.get('IRR_Pct', 0) > 15,
                    "Strong MOIC (> 1.5x)": fund_data.get('MOIC', 0) > 1.5,
                    "Reasonable Fund Size": fund_data.get('Fund_Size_MM', 0) > 100,
                    "Experienced GP": True,  # Would check GP track record
                    "Appropriate Fees (≤ 2.5%)": fund_data.get('Mgmt_Fee_Pct', 3.0) <= 2.5,
                    "Standard Carry (≤ 20%)": fund_data.get('Carry_Pct', 25) <= 20,
                    "Diversified Strategy": True,  # Would check portfolio concentration
                    "Good DPI Progress": fund_data.get('DPI', 0) > 0.5
                }
                
                passed_count = 0
                for criteria, passed in dd_criteria.items():
                    icon = "✅" if passed else "❌"
                    color = "green" if passed else "red"
                    st.markdown(f"<span style='color: {color}'>{icon} {criteria}</span>", unsafe_allow_html=True)
                    if passed:
                        passed_count += 1
                
                # Overall score
                overall_score = (passed_count / len(dd_criteria)) * 100
                st.metric("Overall DD Score", f"{overall_score:.0f}%")
                
                if overall_score >= 80:
                    st.success("🏆 Strong candidate for investment")
                elif overall_score >= 60:
                    st.warning("⚠️ Requires additional due diligence")
                else:
                    st.error("❌ High risk investment")
            
            with col2:
                st.subheader("📊 Peer Comparison")
                
                # Find peer funds (same strategy and similar vintage)
                peers = pe_data[
                    (pe_data['Strategy'] == fund_data.get('Strategy', '')) &
                    (abs(pe_data['Vintage_Year'] - fund_data.get('Vintage_Year', 2020)) <= 2) &
                    (pe_data['Fund_Name'] != selected_dd_fund)
                ]
                
                if len(peers) > 0:
                    # Calculate peer metrics
                    peer_comparison = pd.DataFrame({
                        'Metric': ['IRR (%)', 'MOIC', 'DPI', 'TVPI'],
                        'Selected Fund': [
                            fund_data.get('IRR_Pct', 0),
                            fund_data.get('MOIC', 0),
                            fund_data.get('DPI', 0),
                            fund_data.get('TVPI', 0)
                        ],
                        'Peer Median': [
                            peers['IRR_Pct'].median(),
                            peers['MOIC'].median(),
                            peers['DPI'].median(),
                            peers['TVPI'].median()
                        ],
                        'Peer Average': [
                            peers['IRR_Pct'].mean(),
                            peers['MOIC'].mean(),
                            peers['DPI'].mean(),
                            peers['TVPI'].mean()
                        ]
                    })
                    
                    # Format for display
                    formatted_comparison = peer_comparison.copy()
                    formatted_comparison['Selected Fund'] = formatted_comparison.apply(
                        lambda row: f"{row['Selected Fund']:.1f}%" if row['Metric'] == 'IRR (%)' else f"{row['Selected Fund']:.2f}x", axis=1
                    )
                    formatted_comparison['Peer Median'] = formatted_comparison.apply(
                        lambda row: f"{row['Peer Median']:.1f}%" if row['Metric'] == 'IRR (%)' else f"{row['Peer Median']:.2f}x", axis=1
                    )
                    formatted_comparison['Peer Average'] = formatted_comparison.apply(
                        lambda row: f"{row['Peer Average']:.1f}%" if row['Metric'] == 'IRR (%)' else f"{row['Peer Average']:.2f}x", axis=1
                    )
                    
                    st.dataframe(formatted_comparison[['Metric', 'Selected Fund', 'Peer Median', 'Peer Average']], hide_index=True)
                    
                    # Peer ranking
                    st.write(f"**Peer Group:** {len(peers)} similar funds")
                    
                    irr_rank = (peers['IRR_Pct'] < fund_data.get('IRR_Pct', 0)).sum() + 1
                    st.write(f"**IRR Ranking:** #{irr_rank} out of {len(peers) + 1}")
                    
                else:
                    st.info("No peer funds found for comparison")
    
    # Portfolio-level due diligence
    st.subheader("📈 Portfolio-Level Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Concentration analysis
        if 'GP' in pe_data.columns:
            gp_concentration = pe_data.groupby('GP')['Commitment_MM'].sum().sort_values(ascending=False)
            top_5_gps = gp_concentration.head(5)
            
            st.write("**Top 5 GP Exposures:**")
            for gp, commitment in top_5_gps.items():
                pct = (commitment / pe_data['Commitment_MM'].sum()) * 100
                st.write(f"• {gp}: ${commitment:.0f}M ({pct:.1f}%)")
    
    with col2:
        # Vintage year concentration
        if 'Vintage_Year' in pe_data.columns:
            vintage_concentration = pe_data.groupby('Vintage_Year')['Commitment_MM'].sum().sort_values(ascending=False)
            
            st.write("**Vintage Year Concentration:**")
            for year, commitment in vintage_concentration.head(5).items():
                pct = (commitment / pe_data['Commitment_MM'].sum()) * 100
                st.write(f"• {year}: ${commitment:.0f}M ({pct:.1f}%)")


def show_pe_reports_export(pe_data):
    """PE Reports and Export"""
    
    st.subheader("📋 Reports & Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Generate Reports")
        
        report_type = st.selectbox(
            "Select Report Type",
            [
                "Executive Summary",
                "Detailed Performance Report",
                "Vintage Year Analysis",
                "Due Diligence Report",
                "Cash Flow Analysis",
                "Complete Portfolio Report"
            ],
            key="pe_report_type"
        )
        
        if st.button("🎯 Generate Report", key="pe_generate_report"):
            with st.spinner("Generating PE report..."):
                try:
                    # Create report data based on type
                    report_data = {}
                    
                    if report_type == "Executive Summary":
                        summary_cols = ['Fund_Name', 'GP', 'Strategy', 'Vintage_Year', 'IRR_Pct', 'MOIC', 'DPI', 'TVPI']
                        available_cols = [col for col in summary_cols if col in pe_data.columns]
                        report_data["Executive_Summary"] = pe_data[available_cols]
                    
                    elif report_type == "Detailed Performance Report":
                        perf_cols = ['Fund_Name', 'GP', 'Vintage_Year', 'Fund_Size_MM', 'Commitment_MM', 'Invested_MM', 'IRR_Pct', 'MOIC', 'DPI', 'RVPI', 'TVPI']
                        available_cols = [col for col in perf_cols if col in pe_data.columns]
                        report_data["Performance_Report"] = pe_data[available_cols]
                    
                    elif report_type == "Vintage Year Analysis":
                        if 'Vintage_Year' in pe_data.columns:
                            vintage_summary = pe_data.groupby('Vintage_Year').agg({
                                'IRR_Pct': ['mean', 'std', 'count'],
                                'MOIC': 'mean',
                                'Commitment_MM': 'sum'
                            }).round(2)
                            report_data["Vintage_Analysis"] = vintage_summary
                    
                    else:  # Complete report
                        report_data["Complete_Portfolio"] = pe_data
                    
                    # Generate Excel file
                    buffer = export_data_to_excel(report_data, f"PE_{report_type.replace(' ', '_')}.xlsx")
                    
                    if buffer:
                        st.download_button(
                            label="📥 Download Report",
                            data=buffer,
                            file_name=f"PE_{report_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="pe_report_download"
                        )
                    else:
                        st.error("Failed to generate report")
                        
                except Exception as e:
                    st.error(f"Error generating report: {e}")
    
    with col2:
        st.subheader("💡 Portfolio Insights")
        
        # Generate insights
        insights = []
        
        try:
            # Performance insights
            if 'IRR_Pct' in pe_data.columns:
                top_performer = pe_data.loc[pe_data['IRR_Pct'].idxmax()]
                avg_irr = pe_data['IRR_Pct'].mean()
                insights.append(f"🏆 Top Performer: {top_performer['Fund_Name']} ({top_performer['IRR_Pct']:.1f}% IRR)")
                insights.append(f"📊 Portfolio Average IRR: {avg_irr:.1f}%")
            
            # Strategy insights
            if 'Strategy' in pe_data.columns:
                strategy_performance = pe_data.groupby('Strategy')['IRR_Pct'].mean().sort_values(ascending=False)
                best_strategy = strategy_performance.index[0]
                insights.append(f"🎯 Best Performing Strategy: {best_strategy} ({strategy_performance.iloc[0]:.1f}% avg IRR)")
            
            # Vintage insights
            if 'Vintage_Year' in pe_data.columns:
                vintage_performance = pe_data.groupby('Vintage_Year')['IRR_Pct'].mean().sort_values(ascending=False)
                best_vintage = vintage_performance.index[0]
                insights.append(f"📅 Best Vintage Year: {best_vintage} ({vintage_performance.iloc[0]:.1f}% avg IRR)")
            
            # Portfolio health
            if 'DPI' in pe_data.columns:
                high_dpi_funds = (pe_data['DPI'] > 1.0).sum()
                total_funds = len(pe_data)
                insights.append(f"💰 Funds with DPI > 1.0x: {high_dpi_funds}/{total_funds} ({high_dpi_funds/total_funds*100:.0f}%)")
            
            # Capital deployment
            if all(col in pe_data.columns for col in ['Commitment_MM', 'Invested_MM']):
                total_commitment = pe_data['Commitment_MM'].sum()
                total_invested = pe_data['Invested_MM'].sum()
                deployment_rate = (total_invested / total_commitment) * 100 if total_commitment > 0 else 0
                insights.append(f"📈 Overall Capital Deployment: {deployment_rate:.1f}%")
            
            for insight in insights:
                st.info(insight)
                
        except Exception as e:
            st.warning(f"Could not generate insights: {e}")


# ─── Enhanced Hedge Fund Analysis ──────────────────────────────────────────
def show_complete_hedge_fund_analysis():
    """Complete Hedge Fund Analysis with comprehensive filtering and analytics"""
    
    st.markdown('<h1 class="main-header">🏦 Hedge Fund Performance Analysis</h1>', unsafe_allow_html=True)
    
    # Data Loading Section
    returns_data, mapping_data = load_hedge_fund_data()
    
    if returns_data is not None and mapping_data is not None:
        # Validate data
        returns_issues = validate_data(returns_data, "returns")
        mapping_issues = validate_data(mapping_data, "mapping")
        
        if returns_issues:
            st.error("**Returns Data Issues:**")
            for issue in returns_issues:
                st.error(f"• {issue}")
        
        if mapping_issues:
            st.error("**Mapping Data Issues:**")
            for issue in mapping_issues:
                st.error(f"• {issue}")
        
        if not returns_issues and not mapping_issues:
            # Sidebar Filters
            filtered_funds = create_comprehensive_filters(mapping_data, returns_data)
            
            if filtered_funds:
                # Filter returns data to selected funds
                date_col = 'Date'
                available_return_columns = [col for col in returns_data.columns if col != date_col and col in filtered_funds]
                
                if available_return_columns:
                    filtered_returns = returns_data[[date_col] + available_return_columns].copy()
                    filtered_mapping = mapping_data[mapping_data['Fund Name'].isin(filtered_funds)].copy()
                    
                    st.success(f"✅ Proceeding with analysis of {len(available_return_columns)} funds")
                    
                    # Ensure Date column is datetime
                    try:
                        filtered_returns[date_col] = pd.to_datetime(filtered_returns[date_col])
                    except:
                        st.error("Could not convert Date column to datetime format")
                        return
                    
                    # Main Analysis Tabs
                    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                        "📊 Portfolio Overview",
                        "📈 Performance Analysis", 
                        "🎯 Risk Analytics",
                        "🔍 Factor Analysis",
                        "📉 Drawdown Analysis",
                        "⚡ Stress Testing (Taleb)",
                        "📋 Reports & Export"
                    ])
                    
                    with tab1:
                        show_portfolio_overview(filtered_returns, filtered_mapping)
                    
                    with tab2:
                        show_performance_analysis(filtered_returns, filtered_mapping)
                    
                    with tab3:
                        show_risk_analytics(filtered_returns, filtered_mapping)
                    
                    with tab4:
                        show_factor_analysis(filtered_returns, filtered_mapping)
                    
                    with tab5:
                        show_drawdown_analysis(filtered_returns, filtered_mapping)
                    
                    with tab6:
                        show_stress_scenario_analysis(filtered_returns, filtered_mapping)
                    
                    with tab7:
                        show_reports_export(filtered_returns, filtered_mapping)
                
                else:
                    st.warning("⚠️ No matching funds found in returns data. Please check fund name consistency between Returns and Mapping sheets.")
                    
                    # Show available funds for debugging
                    with st.expander("🔍 Debug: Available Fund Names"):
                        st.write("**In Returns Data:**")
                        returns_funds = [col for col in returns_data.columns if col != 'Date']
                        st.write(returns_funds[:10])  # Show first 10
                        
                        st.write("**In Mapping Data:**")
                        mapping_funds = mapping_data['Fund Name'].tolist()
                        st.write(mapping_funds[:10])  # Show first 10
            else:
                st.warning("⚠️ No funds selected. Please adjust your filters.")
    
    else:
        st.info("👆 Please upload your Excel file to begin analysis.")

    return returns_data, mapping_data




def load_hedge_fund_data():
    """Load and process hedge fund data from Excel file with Returns and Mapping tabs"""
    
    st.subheader("📁 Data Upload & Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "📊 Upload Hedge Fund Excel File",
            type=['xlsx'],
            help="Upload Excel file with 'Returns' and 'Mapping' tabs",
            key="hf_file_upload"
        )
        
        if uploaded_file:
            st.success("✅ File uploaded successfully!")
    
    with col2:
        benchmark = st.selectbox(
            "📈 Select Benchmark",
            ["^GSPC", "^IXIC", "^RUT", "SPY", "QQQ", "AGG"],
            index=0,
            key="hf_benchmark"
        )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Loading hedge fund data..."):
                # Load both sheets
                excel_file = pd.ExcelFile(uploaded_file)
                
                # Check available sheets
                st.info(f"📋 Available sheets: {', '.join(excel_file.sheet_names)}")
                
                returns_data = None
                mapping_data = None
                
                # Load Returns sheet
                if 'Returns' in excel_file.sheet_names:
                    returns_data = pd.read_excel(uploaded_file, sheet_name='Returns')
                    st.success(f"✅ Returns data loaded: {len(returns_data)} periods, {len(returns_data.columns)-1} funds")
                else:
                    st.error("❌ 'Returns' sheet not found")
                
                # Load Mapping sheet
                if 'Mapping' in excel_file.sheet_names:
                    mapping_data = pd.read_excel(uploaded_file, sheet_name='Mapping')
                    st.success(f"✅ Mapping data loaded: {len(mapping_data)} funds with metadata")
                else:
                    st.error("❌ 'Mapping' sheet not found")
                
                return returns_data, mapping_data
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return None, None
    
    return None, None

def create_comprehensive_filters(mapping_data, returns_data):
    """Create comprehensive filtering system based on mapping data"""
    
    with st.sidebar:
        st.header("🎛️ Fund Filters")
        
        # Fund Universe Summary
        st.metric("Total Fund Universe", len(mapping_data))
        
        # Get fund names that exist in both datasets
        returns_funds = set(returns_data.columns) - {'Date'}
        mapping_funds = set(mapping_data['Fund Name'].tolist()) if 'Fund Name' in mapping_data.columns else set()
        available_funds = returns_funds.intersection(mapping_funds)
        
        st.metric("Funds with Both Data & Returns", len(available_funds))
        
        if len(available_funds) == 0:
            st.error("🚨 **No matching funds found between Returns and Mapping sheets!**")
            
            with st.expander("🔍 Debugging Information"):
                st.write("**Sample Returns Fund Names:**")
                st.write(list(returns_funds)[:10])
                st.write("**Sample Mapping Fund Names:**")
                st.write(list(mapping_funds)[:10])
        
        selected_funds = available_funds.copy()  # Start with all available funds
        
        # 1. Market Value Filter
        if '3/31/2025 MV' in mapping_data.columns:
            st.subheader("💰 Market Value")
            mv_col = mapping_data['3/31/2025 MV'].fillna(0)
            if mv_col.max() > 0:
                mv_range = st.slider(
                    "Market Value Range (M)",
                    min_value=float(mv_col.min()),
                    max_value=float(mv_col.max()),
                    value=(float(mv_col.min()), float(mv_col.max())),
                    key="mv_filter"
                )
                mv_filtered = set(mapping_data[
                    (mapping_data['3/31/2025 MV'] >= mv_range[0]) & 
                    (mapping_data['3/31/2025 MV'] <= mv_range[1])
                ]['Fund Name'].tolist())
                selected_funds = selected_funds.intersection(mv_filtered)
        
        # 2. On/Offshore Filter
        if 'On/Offshore' in mapping_data.columns:
            st.subheader("🌍 Domicile")
            domicile_options = mapping_data['On/Offshore'].dropna().unique().tolist()
            if domicile_options:
                selected_domiciles = st.multiselect(
                    "Select Domiciles",
                    domicile_options,
                    default=domicile_options,
                    key="domicile_filter"
                )
                if selected_domiciles:
                    domicile_filtered = set(mapping_data[
                        mapping_data['On/Offshore'].isin(selected_domiciles)
                    ]['Fund Name'].tolist())
                    selected_funds = selected_funds.intersection(domicile_filtered)
        
        # 3. Strategy Filter
        if 'Strategy' in mapping_data.columns:
            st.subheader("🎯 Strategy")
            strategy_options = mapping_data['Strategy'].dropna().unique().tolist()
            if strategy_options:
                selected_strategies = st.multiselect(
                    "Select Strategies",
                    strategy_options,
                    default=strategy_options,
                    key="strategy_filter"
                )
                if selected_strategies:
                    strategy_filtered = set(mapping_data[
                        mapping_data['Strategy'].isin(selected_strategies)
                    ]['Fund Name'].tolist())
                    selected_funds = selected_funds.intersection(strategy_filtered)
        
        # 4. Region Filter
        if 'Region' in mapping_data.columns:
            st.subheader("🗺️ Region")
            region_options = mapping_data['Region'].dropna().unique().tolist()
            if region_options:
                selected_regions = st.multiselect(
                    "Select Regions",
                    region_options,
                    default=region_options,
                    key="region_filter"
                )
                if selected_regions:
                    region_filtered = set(mapping_data[
                        mapping_data['Region'].isin(selected_regions)
                    ]['Fund Name'].tolist())
                    selected_funds = selected_funds.intersection(region_filtered)
        
        # Show filtered results
        st.markdown("---")
        st.subheader("📊 Filter Results")
        st.metric("Funds Selected", len(selected_funds))
        
        if len(selected_funds) > 0:
            with st.expander("👀 View Selected Funds"):
                for fund in sorted(selected_funds):
                    st.write(f"• {fund}")
        
        return list(selected_funds)

def show_portfolio_overview(returns_data, mapping_data):
    """Enhanced portfolio overview with fund composition and high-level metrics"""
    
    st.subheader("📊 Portfolio Composition & Overview")
    
    # High-level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_funds = len(mapping_data)
        st.metric("Total Funds", total_funds)
    
    with col2:
        if '3/31/2025 MV' in mapping_data.columns:
            total_mv = mapping_data['3/31/2025 MV'].sum()
            st.metric("Total AUM", f"${total_mv:,.0f}M")
    
    with col3:
        # Calculate portfolio-level metrics
        fund_metrics = calculate_fund_metrics_batch(returns_data)
        if fund_metrics is not None and not fund_metrics.empty:
            avg_sharpe = fund_metrics['Sharpe'].mean()
            st.metric("Avg Sharpe Ratio", f"{avg_sharpe:.2f}")
    
    with col4:
        if fund_metrics is not None and not fund_metrics.empty:
            avg_return = fund_metrics['Ann_Return'].mean()
            st.metric("Avg Annual Return", f"{avg_return:.1%}")
    
    # Portfolio composition charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Strategy' in mapping_data.columns:
            strategy_counts = mapping_data['Strategy'].value_counts()
            fig_strategy = px.pie(
                values=strategy_counts.values,
                names=strategy_counts.index,
                title="Fund Distribution by Strategy"
            )
            st.plotly_chart(fig_strategy, use_container_width=True)
    
    with col2:
        if 'Region' in mapping_data.columns:
            region_counts = mapping_data['Region'].value_counts()
            fig_region = px.pie(
                values=region_counts.values,
                names=region_counts.index,
                title="Fund Distribution by Region"
            )
            st.plotly_chart(fig_region, use_container_width=True)
    
    # Fund details table
    st.subheader("📋 Fund Details")
    
    # Merge metrics with mapping data
    if fund_metrics is not None and not fund_metrics.empty:
        display_data = mapping_data.set_index('Fund Name').join(fund_metrics, how='inner')
        
        # Select key columns for display
        display_cols = ['Strategy', 'Region', 'Ann_Return', 'Ann_Vol', 'Sharpe', 'Max_DD']
        available_cols = [col for col in display_cols if col in display_data.columns]
        
        if available_cols:
            formatted_data = display_data[available_cols].copy()
            
            # Format percentage columns
            pct_cols = ['Ann_Return', 'Ann_Vol', 'Max_DD']
            for col in pct_cols:
                if col in formatted_data.columns:
                    formatted_data[col] = formatted_data[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
            
            # Format Sharpe ratio
            if 'Sharpe' in formatted_data.columns:
                formatted_data['Sharpe'] = formatted_data['Sharpe'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            
            st.dataframe(formatted_data, use_container_width=True)

def show_performance_analysis(returns_data, mapping_data):
    """Comprehensive performance analysis"""
    
    st.subheader("📈 Performance Analysis")
    
    # Calculate performance metrics for each fund
    fund_metrics = calculate_fund_metrics_batch(returns_data)
    
    if fund_metrics is not None and not fund_metrics.empty:
        # Merge with mapping data for enhanced analysis
        enhanced_metrics = fund_metrics.merge(
            mapping_data.set_index('Fund Name'), 
            left_index=True, 
            right_index=True, 
            how='left'
        )
        
        # Performance comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Return vs Risk scatter
            if all(col in enhanced_metrics.columns for col in ['Ann_Return', 'Ann_Vol']):
                fig_scatter = px.scatter(
                    enhanced_metrics,
                    x='Ann_Vol',
                    y='Ann_Return',
                    color='Strategy' if 'Strategy' in enhanced_metrics.columns else None,
                    hover_name=enhanced_metrics.index,
                    title="Risk-Return Profile",
                    labels={'Ann_Vol': 'Annual Volatility', 'Ann_Return': 'Annual Return'}
                )
                fig_scatter.update_layout(
                    xaxis=dict(tickformat='.1%'),
                    yaxis=dict(tickformat='.1%')
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Sharpe ratio comparison
            if 'Sharpe' in enhanced_metrics.columns:
                top_sharpe = enhanced_metrics.nlargest(10, 'Sharpe')
                fig_sharpe = px.bar(
                    x=top_sharpe['Sharpe'],
                    y=top_sharpe.index,
                    orientation='h',
                    title="Top 10 Funds by Sharpe Ratio"
                )
                st.plotly_chart(fig_sharpe, use_container_width=True)
        
        # Performance by strategy/region
        if 'Strategy' in enhanced_metrics.columns:
            st.subheader("📊 Performance by Strategy")
            
            strategy_perf = enhanced_metrics.groupby('Strategy').agg({
                'Ann_Return': 'mean',
                'Ann_Vol': 'mean',
                'Sharpe': 'mean'
            }).round(4)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig_ret = px.bar(
                    x=strategy_perf.index,
                    y=strategy_perf['Ann_Return'],
                    title="Average Returns by Strategy"
                )
                fig_ret.update_layout(yaxis=dict(tickformat='.1%'))
                st.plotly_chart(fig_ret, use_container_width=True)
            
            with col2:
                fig_vol = px.bar(
                    x=strategy_perf.index,
                    y=strategy_perf['Ann_Vol'],
                    title="Average Volatility by Strategy"
                )
                fig_vol.update_layout(yaxis=dict(tickformat='.1%'))
                st.plotly_chart(fig_vol, use_container_width=True)
            
            with col3:
                fig_sharpe_strat = px.bar(
                    x=strategy_perf.index,
                    y=strategy_perf['Sharpe'],
                    title="Average Sharpe by Strategy"
                )
                st.plotly_chart(fig_sharpe_strat, use_container_width=True)

def show_risk_analytics(returns_data, mapping_data):
    """Comprehensive risk analysis with improved visualizations"""
    
    st.subheader("🎯 Risk Analytics")
    
    fund_metrics = calculate_fund_metrics_batch(returns_data)
    
    if fund_metrics is not None and not fund_metrics.empty:
        # Risk metrics summary
        risk_cols = ['Ann_Vol', 'Max_DD', 'VaR_95', 'CVaR_95', 'Downside_Risk']
        available_risk_cols = [col for col in risk_cols if col in fund_metrics.columns]
        
        if available_risk_cols:
            st.subheader("📊 Risk Metrics Analysis")
            
            # Enhanced visualization with box plots and violin plots
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Risk Overview", "📈 Distribution Analysis", "🎯 Risk Scatter", "📋 Risk Rankings"])
            
            with tab1:
                # Risk Overview with enhanced metrics cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'Ann_Vol' in fund_metrics.columns:
                        vol_data = fund_metrics['Ann_Vol'].dropna()
                        st.metric("Median Volatility", f"{vol_data.median():.2%}")
                        st.metric("Vol Range", f"{vol_data.min():.2%} - {vol_data.max():.2%}")
                
                with col2:
                    if 'Max_DD' in fund_metrics.columns:
                        dd_data = fund_metrics['Max_DD'].dropna()
                        st.metric("Median Max Drawdown", f"{dd_data.median():.2%}")
                        st.metric("Worst Drawdown", f"{dd_data.min():.2%}")
                
                with col3:
                    if 'VaR_95' in fund_metrics.columns:
                        var_data = fund_metrics['VaR_95'].dropna()
                        st.metric("Median VaR (95%)", f"{var_data.median():.2%}")
                        st.metric("Worst VaR", f"{var_data.min():.2%}")
                
                # Combined box plot for all risk metrics
                if len(available_risk_cols) > 0:
                    # Prepare data for box plot
                    risk_data_long = []
                    for col in available_risk_cols:
                        for fund, value in fund_metrics[col].items():
                            if pd.notna(value):
                                risk_data_long.append({
                                    'Fund': fund,
                                    'Metric': col.replace('_', ' '),
                                    'Value': value
                                })
                    
                    risk_df_long = pd.DataFrame(risk_data_long)
                    
                    # Create interactive box plot
                    fig_box = px.box(
                        risk_df_long,
                        x='Metric',
                        y='Value',
                        color='Metric',
                        title='Risk Metrics Distribution (Box Plot)',
                        hover_data=['Fund']
                    )
                    
                    fig_box.update_layout(
                        height=500,
                        showlegend=False,
                        yaxis_title="Value",
                        yaxis=dict(tickformat='.2%')
                    )
                    
                    st.plotly_chart(fig_box, use_container_width=True)
            
            with tab2:
                # Violin plots for better distribution visualization
                st.subheader("📊 Risk Distribution Analysis")
                
                # Select metric for detailed analysis
                selected_metric = st.selectbox(
                    "Select Risk Metric for Detailed Analysis",
                    available_risk_cols,
                    format_func=lambda x: x.replace('_', ' ')
                )
                
                if selected_metric:
                    metric_data = fund_metrics[selected_metric].dropna()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Violin plot with quartiles
                        fig_violin = go.Figure()
                        
                        fig_violin.add_trace(go.Violin(
                            y=metric_data.values,
                            name=selected_metric.replace('_', ' '),
                            box_visible=True,
                            meanline_visible=True,
                            fillcolor='lightblue',
                            opacity=0.6,
                            points='all',
                            jitter=0.5,
                            scalemode='count',
                            text=[f"{fund}: {value:.2%}" for fund, value in metric_data.items()],
                            hovertemplate='%{text}<extra></extra>'
                        ))
                        
                        fig_violin.update_layout(
                            title=f"{selected_metric.replace('_', ' ')} Distribution",
                            yaxis_title="Value",
                            yaxis=dict(tickformat='.2%'),
                            height=500,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_violin, use_container_width=True)
                    
                    with col2:
                        # Empirical CDF
                        sorted_data = np.sort(metric_data.values)
                        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                        
                        fig_cdf = go.Figure()
                        
                        fig_cdf.add_trace(go.Scatter(
                            x=sorted_data,
                            y=cdf,
                            mode='lines',
                            name='Empirical CDF',
                            line=dict(width=3)
                        ))
                        
                        # Add percentile markers
                        percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
                        for p in percentiles:
                            val = np.percentile(metric_data, p * 100)
                            fig_cdf.add_vline(
                                x=val,
                                line_dash="dash",
                                line_color="gray",
                                annotation_text=f"P{int(p*100)}: {val:.2%}",
                                annotation_position="top right"
                            )
                        
                        fig_cdf.update_layout(
                            title=f"Cumulative Distribution - {selected_metric.replace('_', ' ')}",
                            xaxis_title="Value",
                            xaxis=dict(tickformat='.2%'),
                            yaxis_title="Cumulative Probability",
                            height=500
                        )
                        
                        st.plotly_chart(fig_cdf, use_container_width=True)
                    
                    # Distribution statistics
                    st.subheader("📈 Distribution Statistics")
                    
                    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                    
                    with stats_col1:
                        st.metric("Mean", f"{metric_data.mean():.3%}")
                        st.metric("Std Dev", f"{metric_data.std():.3%}")
                    
                    with stats_col2:
                        st.metric("Skewness", f"{skew(metric_data):.3f}")
                        st.metric("Kurtosis", f"{kurtosis(metric_data):.3f}")
                    
                    with stats_col3:
                        st.metric("5th Percentile", f"{np.percentile(metric_data, 5):.3%}")
                        st.metric("95th Percentile", f"{np.percentile(metric_data, 95):.3%}")
                    
                    with stats_col4:
                        # Check for fat tails (kurtosis > 3)
                        kurt_val = kurtosis(metric_data)
                        if kurt_val > 3:
                            st.warning("⚠️ Fat Tails Detected")
                            st.metric("Excess Kurtosis", f"{kurt_val - 3:.2f}")
                        else:
                            st.success("✅ Normal Tails")
                            st.metric("Excess Kurtosis", f"{kurt_val - 3:.2f}")
            
            with tab3:
                # Risk scatter plots
                st.subheader("🎯 Risk-Adjusted Performance")
                
                # Merge with mapping data for strategy colors
                risk_analysis = fund_metrics.merge(
                    mapping_data.set_index('Fund Name'),
                    left_index=True,
                    right_index=True,
                    how='left'
                )
                
                # Volatility vs Drawdown scatter
                if 'Ann_Vol' in risk_analysis.columns and 'Max_DD' in risk_analysis.columns:
                    fig_scatter = px.scatter(
                        risk_analysis,
                        x='Ann_Vol',
                        y='Max_DD',
                        color='Strategy' if 'Strategy' in risk_analysis.columns else None,
                        size='Ann_Return' if 'Ann_Return' in risk_analysis.columns else None,
                        hover_name=risk_analysis.index,
                        hover_data=['Sharpe', 'Sortino'] if all(col in risk_analysis.columns for col in ['Sharpe', 'Sortino']) else None,
                        title="Volatility vs Maximum Drawdown",
                        labels={
                            'Ann_Vol': 'Annual Volatility',
                            'Max_DD': 'Maximum Drawdown'
                        }
                    )
                    
                    # Add diagonal lines for risk ratios
                    x_range = [risk_analysis['Ann_Vol'].min(), risk_analysis['Ann_Vol'].max()]
                    for ratio in [0.5, 1.0, 1.5, 2.0]:
                        fig_scatter.add_trace(go.Scatter(
                            x=x_range,
                            y=[-x * ratio for x in x_range],
                            mode='lines',
                            line=dict(dash='dash', color='gray', width=1),
                            name=f'DD/Vol = {ratio}',
                            showlegend=False
                        ))
                    
                    fig_scatter.update_layout(
                        xaxis=dict(tickformat='.1%'),
                        yaxis=dict(tickformat='.1%'),
                        height=600
                    )
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Risk efficiency frontier
                if all(col in risk_analysis.columns for col in ['Ann_Vol', 'Ann_Return']):
                    # Calculate risk efficiency (return per unit of risk)
                    risk_analysis['Risk_Efficiency'] = risk_analysis['Ann_Return'] / risk_analysis['Ann_Vol']
                    
                    # Sort by volatility for frontier
                    frontier_data = risk_analysis.sort_values('Ann_Vol')
                    
                    # Calculate efficient frontier (cumulative maximum return for each vol level)
                    frontier_data['Efficient_Return'] = frontier_data['Ann_Return'].expanding().max()
                    
                    fig_frontier = go.Figure()
                    
                    # Add all funds
                    fig_frontier.add_trace(go.Scatter(
                        x=risk_analysis['Ann_Vol'],
                        y=risk_analysis['Ann_Return'],
                        mode='markers+text',
                        marker=dict(size=10),
                        text=[f.split()[0][:10] for f in risk_analysis.index],  # Short fund names
                        textposition='top center',
                        name='Funds',
                        hovertext=risk_analysis.index,
                        hovertemplate='<b>%{hovertext}</b><br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                    ))
                    
                    # Add efficient frontier
                    fig_frontier.add_trace(go.Scatter(
                        x=frontier_data['Ann_Vol'],
                        y=frontier_data['Efficient_Return'],
                        mode='lines',
                        line=dict(color='red', width=2),
                        name='Efficient Frontier'
                    ))
                    
                    fig_frontier.update_layout(
                        title="Risk-Return Efficient Frontier",
                        xaxis_title="Annual Volatility",
                        yaxis_title="Annual Return",
                        xaxis=dict(tickformat='.1%'),
                        yaxis=dict(tickformat='.1%'),
                        height=600
                    )
                    
                    st.plotly_chart(fig_frontier, use_container_width=True)
            
            with tab4:
                # Risk ranking with traffic light system
                st.subheader("🏆 Risk Rankings")
                
                # Create comprehensive risk score
                risk_score_data = fund_metrics.copy()
                
                # Normalize risk metrics (lower is better for risk)
                for metric in ['Ann_Vol', 'Max_DD', 'VaR_95', 'CVaR_95', 'Downside_Risk']:
                    if metric in risk_score_data.columns:
                        # Invert and normalize (so higher score = lower risk = better)
                        min_val = risk_score_data[metric].min()
                        max_val = risk_score_data[metric].max()
                        if max_val > min_val:
                            risk_score_data[f'{metric}_Score'] = 1 - ((risk_score_data[metric] - min_val) / (max_val - min_val))
                
                # Calculate composite risk score
                score_cols = [col for col in risk_score_data.columns if col.endswith('_Score')]
                if score_cols:
                    risk_score_data['Composite_Risk_Score'] = risk_score_data[score_cols].mean(axis=1)
                    risk_score_data['Risk_Rank'] = risk_score_data['Composite_Risk_Score'].rank(ascending=False, method='min').astype(int)
                    
                    # Categorize risk levels
                    risk_score_data['Risk_Level'] = pd.qcut(
                        risk_score_data['Composite_Risk_Score'],
                        q=4,
                        labels=['High Risk', 'Medium-High Risk', 'Medium-Low Risk', 'Low Risk']
                    )
                    
                    # Display top and bottom funds
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🟢 Lowest Risk Funds (Top 10)**")
                        low_risk = risk_score_data.nsmallest(10, 'Risk_Rank')[['Risk_Rank', 'Ann_Vol', 'Max_DD', 'Sharpe', 'Risk_Level']]
                        
                        # Format for display
                        low_risk['Ann_Vol'] = low_risk['Ann_Vol'].apply(lambda x: f"{x:.2%}")
                        low_risk['Max_DD'] = low_risk['Max_DD'].apply(lambda x: f"{x:.2%}")
                        low_risk['Sharpe'] = low_risk['Sharpe'].apply(lambda x: f"{x:.2f}")
                        
                        st.dataframe(low_risk, use_container_width=True)
                    
                    with col2:
                        st.write("**🔴 Highest Risk Funds (Bottom 10)**")
                        high_risk = risk_score_data.nlargest(10, 'Risk_Rank')[['Risk_Rank', 'Ann_Vol', 'Max_DD', 'Sharpe', 'Risk_Level']]
                        
                        # Format for display
                        high_risk['Ann_Vol'] = high_risk['Ann_Vol'].apply(lambda x: f"{x:.2%}")
                        high_risk['Max_DD'] = high_risk['Max_DD'].apply(lambda x: f"{x:.2%}")
                        high_risk['Sharpe'] = high_risk['Sharpe'].apply(lambda x: f"{x:.2f}")
                        
                        st.dataframe(high_risk, use_container_width=True)
        
        else:
            st.warning("No risk metrics available for analysis")

# Replace the existing show_factor_analysis function with this:
def show_factor_analysis(returns_data, mapping_data):
    """Enhanced Factor analysis implementation with fund selection, rolling metrics, and Kalman filter"""
    
    st.subheader("🔍 Factor Analysis")
    
    # Fund selection
    fund_columns = [col for col in returns_data.columns if col != 'Date']
    
    if len(fund_columns) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            selected_funds_factor = st.multiselect(
                "Select Funds for Factor Analysis",
                fund_columns,
                default=fund_columns[:min(5, len(fund_columns))],
                key="factor_funds"
            )
        
        with col2:
            # Benchmark selection
            benchmark_options = {
                "S&P 500": "^GSPC",
                "NASDAQ": "^IXIC", 
                "Russell 2000": "^RUT",
                "MSCI World": "URTH",
                "10-Year Treasury": "^TNX",
                "VIX": "^VIX"
            }
            
            selected_benchmark = st.selectbox(
                "Select Benchmark for Analysis",
                list(benchmark_options.keys()),
                key="factor_benchmark"
            )
        
        if selected_funds_factor:
            # Create tabs for different factor analysis methods
            tab1, tab2 = st.tabs(["📊 Single-Factor Analysis", "🎯 Kalman Filter Analysis"])
            
            with tab1:
                show_single_factor_analysis(returns_data, mapping_data, selected_funds_factor, 
                                          selected_benchmark, benchmark_options)
            
            with tab2:
                show_kalman_filter_analysis(returns_data, mapping_data, selected_funds_factor, 
                                          selected_benchmark, benchmark_options)
    
    else:
        st.warning("No funds available for factor analysis")


# Add this new function after show_factor_analysis:
def show_single_factor_analysis(returns_data, mapping_data, selected_funds_factor, 
                               selected_benchmark, benchmark_options):
    """Original single-factor analysis implementation"""
    
    try:
        with st.spinner(f"Downloading {selected_benchmark} data and performing factor analysis..."):
            # Get date range from returns data
            returns_data_clean = returns_data.copy()
            returns_data_clean['Date'] = pd.to_datetime(returns_data_clean['Date'])
            start_date = returns_data_clean['Date'].min()
            end_date = returns_data_clean['Date'].max()
            
            # Download benchmark data
            benchmark_ticker = benchmark_options[selected_benchmark]
            benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)
            
            if benchmark_data.empty:
                st.error(f"Could not download {selected_benchmark} data")
                return
            
            # Handle different yfinance return formats
            if isinstance(benchmark_data.columns, pd.MultiIndex):
                benchmark_data.columns = benchmark_data.columns.droplevel(1)
            
            # Calculate benchmark returns
            if 'Adj Close' in benchmark_data.columns:
                benchmark_prices = benchmark_data['Adj Close']
            elif 'Close' in benchmark_data.columns:
                benchmark_prices = benchmark_data['Close']
            else:
                st.error(f"Could not find price data in {selected_benchmark}")
                return
            
            benchmark_returns = benchmark_prices.pct_change().dropna()
            benchmark_returns.name = f"{selected_benchmark}_Returns"
            
            st.success(f"✅ Downloaded {len(benchmark_returns)} periods of {selected_benchmark} data")
            
            # Perform factor analysis for each selected fund
            factor_results = []
            rolling_data = {}
            
            for fund in selected_funds_factor:
                try:
                    # Get fund returns
                    fund_returns = returns_data[fund].dropna()
                    fund_returns.index = returns_data_clean.loc[fund_returns.index, 'Date']
                    
                    # Align returns with benchmark (handle different start dates)
                    aligned_data = pd.concat([fund_returns, benchmark_returns], axis=1, join='inner')
                    
                    if len(aligned_data) < 24:  # Need at least 24 months
                        st.warning(f"⚠️ {fund}: Insufficient data ({len(aligned_data)} periods)")
                        continue
                    
                    fund_aligned = aligned_data.iloc[:, 0]
                    bench_aligned = aligned_data.iloc[:, 1]
                    
                    # Calculate factor metrics
                    factor_metrics = calculate_factor_metrics(fund_aligned, bench_aligned, fund)
                    factor_results.append(factor_metrics)
                    
                    # Calculate rolling metrics for charts
                    rolling_metrics = calculate_rolling_factor_metrics(fund_aligned, bench_aligned, fund)
                    rolling_data[fund] = rolling_metrics
                    
                except Exception as e:
                    st.warning(f"⚠️ Error analyzing {fund}: {e}")
                    continue
            
            if factor_results:
                # Display results
                display_factor_results(factor_results, selected_benchmark)
                
                # Display rolling charts
                if rolling_data:
                    display_rolling_charts(rolling_data, selected_benchmark)
            
            else:
                st.warning("Could not calculate factor metrics for any selected funds")
    
    except Exception as e:
        st.error(f"Error in factor analysis: {e}")
        st.info("💡 Try selecting a different benchmark or check your internet connection")


# Add these new Kalman filter functions:
def show_kalman_filter_analysis(returns_data, mapping_data, selected_funds_factor, 
                               selected_benchmark, benchmark_options):
    """Kalman filter implementation for dynamic factor analysis"""
    
    st.subheader("🎯 Kalman Filter Factor Analysis")
    
    show_info_button(
        "kalman_filter_info",
        "Understanding Kalman Filter Analysis",
        """
        **Kalman Filter for Dynamic Factor Analysis**
        
        The Kalman filter provides time-varying estimates of:
        - **Dynamic Alpha**: How manager skill varies over time
        - **Dynamic Beta**: How market exposure changes
        - **Uncertainty Bands**: Confidence intervals for estimates
        
        **Key Difference: Static OLS vs Dynamic Kalman Filter**
        
        📊 **Single-Factor OLS (Traditional Approach):**
        - Assumes alpha and beta are CONSTANT over the entire period
        - Provides one alpha and one beta for all time periods
        - Cannot detect changes in strategy or market exposure
        - Example: Beta = 0.8 for the entire 5-year period
        
        🎯 **Kalman Filter (Dynamic Approach):**
        - Allows alpha and beta to CHANGE over time
        - Provides different estimates for each time period
        - Detects regime changes and strategy drift
        - Example: Beta starts at 0.6, increases to 1.2 during bull markets, drops to 0.4 during crises
        
        **Why This Matters:**
        - Hedge funds often change their strategies over time
        - Market exposure can vary with market conditions
        - Manager skill may improve or deteriorate
        - Static analysis can miss these crucial changes
        
        **Real-World Example:**
        A fund might appear to have a beta of 0.8 over 5 years using OLS.
        But Kalman filter might reveal:
        - Years 1-2: Beta = 0.4 (defensive positioning)
        - Years 3-4: Beta = 1.2 (aggressive growth phase)
        - Year 5: Beta = 0.6 (risk reduction)
        
        The average is still 0.8, but the dynamic story is much more informative!
        
        **Key Parameters:**
        - Process noise: How quickly parameters can change
        - Measurement noise: Observation uncertainty
        - Initial uncertainty: Starting confidence in estimates
        """
    )
    
    
    # Kalman filter parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        process_noise = st.slider(
            "Process Noise (σ²)",
            min_value=0.0001,
            max_value=0.01,
            value=0.001,
            step=0.0001,
            format="%.4f",
            help="Higher values allow parameters to change more quickly"
        )
    
    with col2:
        measurement_noise = st.slider(
            "Measurement Noise (R)",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.3f",
            help="Higher values indicate more observation uncertainty"
        )
    
    with col3:
        initial_uncertainty = st.slider(
            "Initial Uncertainty (P₀)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Higher values indicate less confidence in initial estimates"
        )
    
    try:
        with st.spinner(f"Running Kalman filter analysis..."):
            # Get benchmark data
            returns_data_clean = returns_data.copy()
            returns_data_clean['Date'] = pd.to_datetime(returns_data_clean['Date'])
            start_date = returns_data_clean['Date'].min()
            end_date = returns_data_clean['Date'].max()
            
            # Download benchmark data
            benchmark_ticker = benchmark_options[selected_benchmark]
            benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)
            
            if benchmark_data.empty:
                st.error(f"Could not download {selected_benchmark} data")
                return
            
            # Process benchmark data
            if isinstance(benchmark_data.columns, pd.MultiIndex):
                benchmark_data.columns = benchmark_data.columns.droplevel(1)
            
            if 'Adj Close' in benchmark_data.columns:
                benchmark_prices = benchmark_data['Adj Close']
            elif 'Close' in benchmark_data.columns:
                benchmark_prices = benchmark_data['Close']
            else:
                st.error(f"Could not find price data in {selected_benchmark}")
                return
            
            benchmark_returns = benchmark_prices.pct_change().dropna()
            
            # Run Kalman filter for each fund
            kalman_results = {}
            
            for fund in selected_funds_factor:
                try:
                    # Get fund returns
                    fund_returns = returns_data[fund].dropna()
                    fund_returns.index = returns_data_clean.loc[fund_returns.index, 'Date']
                    
                    # Align with benchmark
                    aligned_data = pd.concat([fund_returns, benchmark_returns], axis=1, join='inner').dropna()
                    
                    if len(aligned_data) < 24:
                        st.warning(f"⚠️ {fund}: Insufficient data")
                        continue
                    
                    # Run Kalman filter
                    kf_results = run_kalman_filter(
                        aligned_data.iloc[:, 0].values,  # Fund returns
                        aligned_data.iloc[:, 1].values,  # Benchmark returns
                        process_noise,
                        measurement_noise,
                        initial_uncertainty
                    )
                    
                    kf_results['dates'] = aligned_data.index
                    kalman_results[fund] = kf_results
                    
                except Exception as e:
                    st.warning(f"Error analyzing {fund}: {e}")
                    continue
            
            if kalman_results:
                display_kalman_results(kalman_results, selected_benchmark)
            else:
                st.warning("No results to display")
                
    except Exception as e:
        st.error(f"Error in Kalman filter analysis: {e}")


def run_kalman_filter(fund_returns, benchmark_returns, process_noise, measurement_noise, initial_uncertainty):
    """
    Run Kalman filter to estimate time-varying alpha and beta
    
    State vector: [alpha, beta]
    Measurement equation: r_fund = alpha + beta * r_benchmark + noise
    """
    
    n_periods = len(fund_returns)
    
    # Initialize state and covariance
    state = np.array([0.0, 1.0])  # [alpha, beta]
    P = np.eye(2) * initial_uncertainty  # Initial uncertainty
    
    # Process and measurement noise
    Q = np.eye(2) * process_noise  # Process noise covariance
    R = measurement_noise  # Measurement noise variance
    
    # Storage for results
    alphas = []
    betas = []
    alpha_std = []
    beta_std = []
    innovations = []
    
    # Run Kalman filter
    for t in range(n_periods):
        # Prediction step (no dynamics, so state prediction is unchanged)
        state_pred = state
        P_pred = P + Q
        
        # Measurement update
        H = np.array([1.0, benchmark_returns[t]])  # Measurement matrix
        y = fund_returns[t]  # Actual measurement
        y_pred = H @ state_pred  # Predicted measurement
        
        # Innovation
        innovation = y - y_pred
        innovations.append(innovation)
        
        # Innovation covariance
        S = H @ P_pred @ H.T + R
        
        # Kalman gain
        K = P_pred @ H.T / S
        
        # State update
        state = state_pred + K * innovation
        
        # Covariance update
        P = (np.eye(2) - np.outer(K, H)) @ P_pred
        
        # Store results
        alphas.append(state[0] * 12)  # Annualize alpha
        betas.append(state[1])
        alpha_std.append(np.sqrt(P[0, 0]) * 12)  # Annualized std
        beta_std.append(np.sqrt(P[1, 1]))
    
    return {
        'alpha': np.array(alphas),
        'beta': np.array(betas),
        'alpha_std': np.array(alpha_std),
        'beta_std': np.array(beta_std),
        'innovations': np.array(innovations)
    }


def display_kalman_results(kalman_results, benchmark_name):
    """Display Kalman filter results"""
    
    st.subheader(f"📊 Dynamic Factor Estimates vs {benchmark_name}")
    
    # Add explanation box for the metrics
    show_info_button(
        "kalman_metrics_explanation",
        "Understanding the Dynamic Factor Metrics",
        """
        **What Each Metric Tells You:**
        
        📈 **Final Alpha**
        - The fund's excess return (alpha) at the END of the analysis period
        - Positive = outperforming the benchmark after adjusting for market risk
        - Negative = underperforming
        - Example: Final Alpha = 0.05 means the fund is currently generating 5% annual excess return
        
        📊 **Final Beta**
        - The fund's market sensitivity at the END of the analysis period
        - Beta = 1.0: Moves exactly with the market
        - Beta > 1.0: Amplifies market moves (more aggressive)
        - Beta < 1.0: Dampens market moves (more defensive)
        - Example: Final Beta = 1.2 means a 10% market move causes a 12% fund move
        
        📉 **Avg Alpha**
        - The AVERAGE excess return over the entire period
        - Shows the fund's typical performance level
        - Compare to Final Alpha to see if performance is improving or deteriorating
        
        📊 **Avg Beta**
        - The AVERAGE market exposure over the entire period
        - Shows the fund's typical risk level
        - Compare to Final Beta to see if the fund is becoming more/less aggressive
        
        🌊 **Alpha Volatility**
        - How much the alpha CHANGES over time
        - Low volatility = Consistent performance (good!)
        - High volatility = Erratic performance (concerning)
        - Example: High alpha volatility might indicate the manager is struggling
        
        🌊 **Beta Volatility**
        - How much the market exposure CHANGES over time
        - Low volatility = Consistent strategy (good for style purity)
        - High volatility = Frequent strategy changes (may indicate drift)
        - Example: High beta volatility might mean the fund is market timing
        
        **How to Interpret:**
        - Compare Final vs Average values to identify trends
        - Low volatility metrics generally indicate more consistent management
        - Watch for divergence between current (Final) and historical (Avg) values
        """
    )
    
    # Summary statistics
    summary_data = []
    for fund, results in kalman_results.items():
        summary_data.append({
            'Fund': fund,
            'Final Alpha': results['alpha'][-1],
            'Final Beta': results['beta'][-1],
            'Avg Alpha': np.mean(results['alpha']),
            'Avg Beta': np.mean(results['beta']),
            'Alpha Volatility': np.std(results['alpha']),
            'Beta Volatility': np.std(results['beta'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Format for display
    for col in ['Final Alpha', 'Avg Alpha', 'Alpha Volatility']:
        summary_df[col] = summary_df[col].apply(lambda x: f"{x:.3f}")
    for col in ['Final Beta', 'Avg Beta', 'Beta Volatility']:
        summary_df[col] = summary_df[col].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Add interpretation helper
    st.markdown("---")
    st.markdown("**🔍 Quick Interpretation Guide:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Alpha Trends:**
        - ↗️ Final > Avg: Improving
        - ↘️ Final < Avg: Deteriorating
        - ➡️ Final ≈ Avg: Stable
        """)
    
    with col2:
        st.markdown("""
        **Beta Trends:**
        - ↗️ Final > Avg: Getting aggressive
        - ↘️ Final < Avg: Getting defensive
        - ➡️ Final ≈ Avg: Stable exposure
        """)
    
    with col3:
        st.markdown("""
        **Volatility Levels:**
        - 🟢 < 0.05: Very stable
        - 🟡 0.05-0.15: Moderate
        - 🔴 > 0.15: High variability
        """)
    
    # Detailed visualization for selected fund
    selected_fund = st.selectbox(
        "Select fund for detailed Kalman filter visualization",
        list(kalman_results.keys()),
        key="kalman_detail_fund"
    )
    
    if selected_fund:
        results = kalman_results[selected_fund]
        dates = results['dates']
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Dynamic Alpha (Annualized)',
                'Dynamic Beta',
                'Innovation Process'
            ),
            vertical_spacing=0.1
        )
        
        # Alpha plot with confidence bands
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=results['alpha'],
                mode='lines',
                name='Alpha',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add confidence bands
        upper_alpha = results['alpha'] + 2 * results['alpha_std']
        lower_alpha = results['alpha'] - 2 * results['alpha_std']
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=upper_alpha,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=lower_alpha,
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(0,100,200,0.2)',
                fill='tonexty',
                name='95% CI',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Beta plot with confidence bands
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=results['beta'],
                mode='lines',
                name='Beta',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        upper_beta = results['beta'] + 2 * results['beta_std']
        lower_beta = results['beta'] - 2 * results['beta_std']
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=upper_beta,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=lower_beta,
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(200,100,0,0.2)',
                fill='tonexty',
                name='95% CI',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Innovation plot
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=results['innovations'],
                mode='lines',
                name='Innovations',
                line=dict(color='green', width=1)
            ),
            row=3, col=1
        )
        
        # Add horizontal lines
        fig.add_hline(y=0, row=1, col=1, line_dash="dash", line_color="gray")
        fig.add_hline(y=1, row=2, col=1, line_dash="dash", line_color="gray")
        fig.add_hline(y=0, row=3, col=1, line_dash="dash", line_color="gray")
        
        # Update layout
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Alpha", row=1, col=1)
        fig.update_yaxes(title_text="Beta", row=2, col=1)
        fig.update_yaxes(title_text="Innovation", row=3, col=1)
        
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text=f"Kalman Filter Results: {selected_fund}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Parameter Evolution")
            
            # Check for regime changes
            alpha_change = results['alpha'][-1] - results['alpha'][0]
            beta_change = results['beta'][-1] - results['beta'][0]
            
            if abs(alpha_change) > 0.02:  # 2% annualized
                direction = "increased" if alpha_change > 0 else "decreased"
                st.info(f"Alpha has {direction} by {abs(alpha_change):.3f} over the period")
            
            if abs(beta_change) > 0.2:
                direction = "increased" if beta_change > 0 else "decreased"
                st.info(f"Beta has {direction} by {abs(beta_change):.3f} over the period")
            
            # Stability analysis
            alpha_vol = np.std(results['alpha'])
            beta_vol = np.std(results['beta'])
            
            st.metric("Alpha Stability", f"{1/alpha_vol:.1f}" if alpha_vol > 0 else "∞")
            st.metric("Beta Stability", f"{1/beta_vol:.1f}" if beta_vol > 0 else "∞")
        
# Replace the "with col2:" section in display_kalman_results with this enhanced version:

        with col2:
            st.subheader("🎯 Model Diagnostics")
            
            # Add explanation for stability metrics
            show_info_button(
                "stability_metrics_info",
                "Understanding Stability Metrics",
                """
                **Alpha & Beta Stability Scores**
                
                These metrics measure how CONSISTENT the parameters are over time.
                Higher scores = More stable = Better!
                
                📊 **Alpha Stability**
                - Measures consistency of excess returns
                - Score = 1 / (Alpha Volatility)
                - **Interpretation:**
                  - 🟢 > 20: Excellent - Very consistent performance
                  - 🟢 10-20: Good - Reasonably stable
                  - 🟡 5-10: Moderate - Some variability
                  - 🔴 < 5: Poor - Highly unstable performance
                
                📈 **Beta Stability**
                - Measures consistency of market exposure
                - Score = 1 / (Beta Volatility)
                - **Interpretation:**
                  - 🟢 > 20: Excellent - Very consistent strategy
                  - 🟢 10-20: Good - Stable approach
                  - 🟡 5-10: Moderate - Some style drift
                  - 🔴 < 5: Poor - Frequent strategy changes
                
                **Why Stability Matters:**
                - High stability suggests disciplined, consistent management
                - Low stability may indicate:
                  - Strategy confusion
                  - Market timing attempts
                  - Reactive rather than proactive management
                
                **Example:**
                - Alpha Stability = 25: Manager delivers consistent excess returns
                - Alpha Stability = 3: Manager's performance is erratic and unpredictable
                """
            )
            
            # Innovation analysis
            innovations = results['innovations']
            
            # Add explanation for innovation metrics
            show_info_button(
                "innovation_metrics_info",
                "Understanding Innovation Metrics",
                """
                **What are Innovations?**
                
                Innovations are the "surprises" or "errors" in the model - the difference between 
                what the model predicted and what actually happened.
                
                📊 **Innovation Mean**
                - Average prediction error
                - Should be close to ZERO
                - **Interpretation:**
                  - 🟢 -0.001 to 0.001: Excellent - No systematic bias
                  - 🟡 -0.005 to -0.001 or 0.001 to 0.005: Small bias
                  - 🔴 Beyond ±0.005: Model may be mis-specified
                
                📈 **Innovation Std Dev**
                - Volatility of prediction errors
                - Lower is better (more predictable)
                - **Interpretation:**
                  - 🟢 < 0.02: Very predictable
                  - 🟡 0.02-0.05: Moderately predictable
                  - 🔴 > 0.05: Highly unpredictable
                
                **What This Tells You:**
                - Mean ≈ 0: Model is unbiased (good!)
                - Mean ≠ 0: Systematic over/under-prediction
                - Low Std Dev: Fund follows model closely
                - High Std Dev: Fund behavior is erratic
                
                **Red Flags:**
                - Large positive mean: Model underestimates returns
                - Large negative mean: Model overestimates returns
                - Very high std dev: Fund doesn't follow normal patterns
                """
            )
            
            # Calculate and display stability metrics with interpretation
            alpha_vol = np.std(results['alpha'])
            beta_vol = np.std(results['beta'])
            
            # Alpha stability with color coding
            alpha_stability = 1/alpha_vol if alpha_vol > 0 else float('inf')
            alpha_color = "🟢" if alpha_stability > 20 else "🟢" if alpha_stability > 10 else "🟡" if alpha_stability > 5 else "🔴"
            
            # Beta stability with color coding  
            beta_stability = 1/beta_vol if beta_vol > 0 else float('inf')
            beta_color = "🟢" if beta_stability > 20 else "🟢" if beta_stability > 10 else "🟡" if beta_stability > 5 else "🔴"
            
            col2a, col2b = st.columns(2)
            
            with col2a:
                st.metric(
                    f"{alpha_color} Alpha Stability", 
                    f"{alpha_stability:.1f}" if alpha_stability != float('inf') else "∞",
                    help="Higher is better. >20 excellent, 10-20 good, 5-10 moderate, <5 poor"
                )
            
            with col2b:
                st.metric(
                    f"{beta_color} Beta Stability", 
                    f"{beta_stability:.1f}" if beta_stability != float('inf') else "∞",
                    help="Higher is better. >20 excellent, 10-20 good, 5-10 moderate, <5 poor"
                )
            
            # Innovation statistics with interpretation
            innovation_mean = np.mean(innovations)
            innovation_std = np.std(innovations)
            
            # Color coding for innovation mean
            mean_color = "🟢" if abs(innovation_mean) < 0.001 else "🟡" if abs(innovation_mean) < 0.005 else "🔴"
            
            # Color coding for innovation std
            std_color = "🟢" if innovation_std < 0.02 else "🟡" if innovation_std < 0.05 else "🔴"
            
            col2c, col2d = st.columns(2)
            
            with col2c:
                st.metric(
                    f"{mean_color} Innovation Mean", 
                    f"{innovation_mean:.4f}",
                    help="Should be near 0. ±0.001 excellent, ±0.005 acceptable, beyond is concerning"
                )
            
            with col2d:
                st.metric(
                    f"{std_color} Innovation Std Dev", 
                    f"{innovation_std:.4f}",
                    help="Lower is better. <0.02 very predictable, 0.02-0.05 moderate, >0.05 unpredictable"
                )
            
            # Check for autocorrelation in innovations
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_test = acorr_ljungbox(innovations, lags=10, return_df=True)
            
            if (lb_test['lb_pvalue'] < 0.05).any():
                st.warning("⚠️ Autocorrelation detected in innovations")
            else:
                st.success("✅ No significant autocorrelation")
            
            # CUSUM test for structural breaks
            cumsum_innovations = np.cumsum(innovations) / np.sqrt(len(innovations))
            max_cusum = np.max(np.abs(cumsum_innovations))
            
            if max_cusum > 1.358:  # 5% critical value
                st.warning("⚠️ Potential structural break detected")
            else:
                st.success("✅ No structural breaks detected")               
                

def display_factor_results(factor_results, benchmark_name):
    """Display factor analysis results table with statistical significance highlighting"""
    
    st.subheader(f"📊 Factor Analysis Results vs {benchmark_name}")
    
    # Create DataFrame
    factor_df = pd.DataFrame(factor_results)
    
    if factor_df.empty:
        st.warning("No factor analysis results to display")
        return
    
    # Format for display
    display_df = factor_df.copy()
    
    # Create formatted columns with significance indicators
    def format_with_significance(value, pvalue, format_str="{:.3f}"):
        if pd.isna(value) or pd.isna(pvalue):
            return "N/A"
        
        formatted_value = format_str.format(value)
        
        if pvalue < 0.01:
            return f"{formatted_value} ***"
        elif pvalue < 0.05:
            return f"{formatted_value} **"
        elif pvalue < 0.10:
            return f"{formatted_value} *"
        else:
            return formatted_value
    
    # Apply formatting
    display_df['Jensen Alpha'] = display_df.apply(
        lambda row: format_with_significance(row['Alpha'], row['Alpha_PValue'], "{:.3f}"), axis=1
    )
    
    display_df['Beta'] = display_df.apply(
        lambda row: format_with_significance(row['Beta'], row['Beta_PValue'], "{:.3f}"), axis=1
    )
    
    display_df['Bull Beta'] = display_df['Bull_Beta'].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
    )
    
    display_df['Bear Beta'] = display_df['Bear_Beta'].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
    )
    
    display_df['R-Squared'] = display_df['R_Squared'].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
    )
    
    # Select columns for display
    final_display = display_df[['Fund', 'Jensen Alpha', 'Beta', 'Bull Beta', 'Bear Beta', 'R-Squared', 'Observations']]
    
    st.dataframe(final_display, use_container_width=True, hide_index=True)
    
    # Add legend
    st.caption("Statistical Significance: *** p<0.01, ** p<0.05, * p<0.10")
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_alpha = factor_df['Alpha'].mean()
        st.metric("Average Alpha", f"{avg_alpha:.3f}")
    
    with col2:
        avg_beta = factor_df['Beta'].mean()
        st.metric("Average Beta", f"{avg_beta:.3f}")
    
    with col3:
        avg_rsq = factor_df['R_Squared'].mean()
        st.metric("Average R²", f"{avg_rsq:.3f}")

def display_rolling_charts(rolling_data, benchmark_name):
    """Display rolling alpha and beta charts"""
    
    st.subheader(f"📈 Rolling Factor Analysis vs {benchmark_name}")
    
    # Create tabs for different time periods
    if '12M' in list(rolling_data.values())[0] and '36M' in list(rolling_data.values())[0]:
        tab1, tab2 = st.tabs(["📊 12-Month Rolling", "📊 36-Month Rolling"])
        
        for tab, window in zip([tab1, tab2], ['12M', '36M']):
            with tab:
                # Alpha chart
                fig_alpha = go.Figure()
                
                for fund, data in rolling_data.items():
                    if window in data:
                        fig_alpha.add_trace(go.Scatter(
                            x=data[window]['dates'],
                            y=data[window]['alpha'],
                            mode='lines',
                            name=f"{fund}",
                            line=dict(width=2)
                        ))
                
                fig_alpha.add_hline(y=0, line_dash="dash", line_color="gray", 
                                   annotation_text="Zero Alpha")
                
                fig_alpha.update_layout(
                    title=f"{window} Rolling Alpha",
                    xaxis_title="Date",
                    yaxis_title="Alpha",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_alpha, use_container_width=True)
                
                # Beta chart
                fig_beta = go.Figure()
                
                for fund, data in rolling_data.items():
                    if window in data:
                        fig_beta.add_trace(go.Scatter(
                            x=data[window]['dates'],
                            y=data[window]['beta'],
                            mode='lines',
                            name=f"{fund}",
                            line=dict(width=2)
                        ))
                
                fig_beta.add_hline(y=1, line_dash="dash", line_color="gray", 
                                  annotation_text="Beta = 1")
                
                fig_beta.update_layout(
                    title=f"{window} Rolling Beta",
                    xaxis_title="Date",
                    yaxis_title="Beta",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_beta, use_container_width=True)
    
    else:
        st.warning("Insufficient data for rolling analysis")

def show_drawdown_analysis(returns_data, mapping_data):
    """Enhanced drawdown analysis implementation with benchmark comparison"""
    
    st.subheader("📉 Drawdown Analysis")
    
    # Select funds for analysis
    fund_columns = [col for col in returns_data.columns if col != 'Date']
    
    if len(fund_columns) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            selected_funds_dd = st.multiselect(
                "Select Funds for Drawdown Analysis",
                fund_columns,
                default=fund_columns[:min(5, len(fund_columns))],  # Default to first 5 funds
                key="drawdown_funds"
            )
        
        with col2:
            # Benchmark selection
            benchmark_options = {
                "S&P 500": "^GSPC",
                "NASDAQ": "^IXIC", 
                "Russell 2000": "^RUT",
                "MSCI World": "URTH",
                "10-Year Treasury": "^TNX",
                "VIX": "^VIX",
                "No Benchmark": None
            }
            
            selected_benchmark = st.selectbox(
                "Select Benchmark for Comparison",
                list(benchmark_options.keys()),
                index=0,  # Default to S&P 500
                key="drawdown_benchmark"
            )
        
        if selected_funds_dd:
            benchmark_returns = None
            benchmark_name = None
            
            # Download benchmark data if selected
            if selected_benchmark != "No Benchmark":
                try:
                    with st.spinner(f"Downloading {selected_benchmark} data..."):
                        # Get date range from returns data
                        returns_data_clean = returns_data.copy()
                        returns_data_clean['Date'] = pd.to_datetime(returns_data_clean['Date'])
                        start_date = returns_data_clean['Date'].min()
                        end_date = returns_data_clean['Date'].max()
                        
                        # Download benchmark data
                        benchmark_ticker = benchmark_options[selected_benchmark]
                        benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)
                        
                        if not benchmark_data.empty:
                            # Handle different yfinance return formats
                            if isinstance(benchmark_data.columns, pd.MultiIndex):
                                benchmark_data.columns = benchmark_data.columns.droplevel(1)
                            
                            # Calculate benchmark returns
                            if 'Adj Close' in benchmark_data.columns:
                                benchmark_prices = benchmark_data['Adj Close']
                            elif 'Close' in benchmark_data.columns:
                                benchmark_prices = benchmark_data['Close']
                            else:
                                st.warning(f"Could not find price data for {selected_benchmark}")
                                benchmark_returns = None
                            
                            if benchmark_prices is not None:
                                benchmark_returns = benchmark_prices.pct_change().dropna()
                                benchmark_name = selected_benchmark
                                st.success(f"✅ Downloaded {len(benchmark_returns)} periods of {selected_benchmark} data")
                        else:
                            st.warning(f"Could not download {selected_benchmark} data")
                            
                except Exception as e:
                    st.warning(f"Error downloading benchmark: {e}")
                    benchmark_returns = None
            
            # Calculate drawdowns for selected funds and benchmark
            fig_dd = go.Figure()
            
            fund_dd_data = {}
            all_dates = []
            
            # Process each fund with proper date alignment
            for fund in selected_funds_dd:
                try:
                    fund_returns = returns_data[fund].dropna()
                    if len(fund_returns) > 0:
                        # Get corresponding dates for this fund
                        fund_indices = fund_returns.index
                        fund_dates = returns_data_clean.loc[fund_indices, 'Date']
                        
                        # Calculate drawdown series
                        dd_series = calculate_drawdown_series(fund_returns)
                        
                        if len(dd_series) > 0:
                            # Align drawdown with dates
                            dd_with_dates = pd.Series(dd_series.values, index=fund_dates)
                            fund_dd_data[fund] = dd_with_dates
                            all_dates.extend(fund_dates.tolist())
                            
                            fig_dd.add_trace(go.Scatter(
                                x=fund_dates,
                                y=dd_series.values,
                                mode='lines',
                                name=fund,
                                line=dict(width=2),
                                hovertemplate=f'<b>{fund}</b><br>Date: %{{x}}<br>Drawdown: %{{y:.2%}}<extra></extra>'
                            ))
                except Exception as e:
                    st.warning(f"Error calculating drawdown for {fund}: {e}")
                    continue
            
            # Add benchmark drawdown if available, aligned to fund date range
            benchmark_dd_data = None
            if benchmark_returns is not None and all_dates:
                try:
                    # Get the full date range from all selected funds
                    min_date = min(all_dates)
                    max_date = max(all_dates)
                    
                    # Filter benchmark returns to match fund date range
                    benchmark_returns.index = pd.to_datetime(benchmark_returns.index)
                    benchmark_filtered = benchmark_returns[(benchmark_returns.index >= min_date) & 
                                                         (benchmark_returns.index <= max_date)]
                    
                    if len(benchmark_filtered) > 0:
                        benchmark_dd_series = calculate_drawdown_series(benchmark_filtered)
                        
                        if len(benchmark_dd_series) > 0:
                            # Align benchmark drawdown with its dates
                            benchmark_dates = benchmark_filtered.index[len(benchmark_filtered) - len(benchmark_dd_series):]
                            benchmark_dd_data = pd.Series(benchmark_dd_series.values, index=benchmark_dates)
                            
                            fig_dd.add_trace(go.Scatter(
                                x=benchmark_dates,
                                y=benchmark_dd_series.values,
                                mode='lines',
                                name=f"{benchmark_name} (Benchmark)",
                                line=dict(width=3, dash='dash', color='red'),
                                hovertemplate=f'<b>{benchmark_name}</b><br>Date: %{{x}}<br>Drawdown: %{{y:.2%}}<extra></extra>'
                            ))
                except Exception as e:
                    st.warning(f"Error calculating benchmark drawdown: {e}")
            
            # Update chart layout with proper date formatting
            chart_title = "Drawdown Analysis - Underwater Chart"
            if benchmark_name:
                chart_title += f" vs {benchmark_name}"
            
            fig_dd.update_layout(
                title=chart_title,
                xaxis_title="Date",
                yaxis_title="Drawdown",
                yaxis=dict(tickformat='.1%'),
                xaxis=dict(
                    type='date',
                    tickformat='%Y-%m',
                    dtick='M6'  # Show ticks every 6 months
                ),
                height=600,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_dd, use_container_width=True)
            
            # Enhanced drawdown statistics with benchmark comparison
            st.subheader("📊 Drawdown Statistics")
            
            dd_stats = []
            
            # Calculate fund statistics
            for fund in selected_funds_dd:
                if fund in fund_dd_data:
                    try:
                        dd_series = fund_dd_data[fund]
                        max_dd = dd_series.min()
                        avg_dd = dd_series.mean()
                        dd_periods = (dd_series < -0.01).sum()  # Periods with >1% drawdown
                        
                        # Calculate recovery metrics
                        recovery_periods = calculate_recovery_time(dd_series)
                        
                        dd_stats.append({
                            'Fund/Benchmark': fund,
                            'Type': 'Fund',
                            'Max Drawdown': max_dd,
                            'Avg Drawdown': avg_dd,
                            'DD Periods (>1%)': dd_periods,
                            'Avg Recovery (Periods)': recovery_periods
                        })
                    except Exception as e:
                        continue
            
            # Add benchmark statistics
            if benchmark_dd_data is not None:
                try:
                    max_dd_bench = benchmark_dd_data.min()
                    avg_dd_bench = benchmark_dd_data.mean()
                    dd_periods_bench = (benchmark_dd_data < -0.01).sum()
                    recovery_periods_bench = calculate_recovery_time(benchmark_dd_data)
                    
                    dd_stats.append({
                        'Fund/Benchmark': f"{benchmark_name} (Benchmark)",
                        'Type': 'Benchmark',
                        'Max Drawdown': max_dd_bench,
                        'Avg Drawdown': avg_dd_bench,
                        'DD Periods (>1%)': dd_periods_bench,
                        'Avg Recovery (Periods)': recovery_periods_bench
                    })
                except Exception as e:
                    st.warning(f"Error calculating benchmark statistics: {e}")
            
            if dd_stats:
                dd_df = pd.DataFrame(dd_stats)
                
                # Format for display
                formatted_dd = dd_df.copy()
                formatted_dd['Max Drawdown'] = formatted_dd['Max Drawdown'].apply(lambda x: f"{x:.2%}")
                formatted_dd['Avg Drawdown'] = formatted_dd['Avg Drawdown'].apply(lambda x: f"{x:.2%}")
                formatted_dd['Avg Recovery (Periods)'] = formatted_dd['Avg Recovery (Periods)'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
                
                # Style the dataframe to highlight benchmark
                def highlight_benchmark(row):
                    if row['Type'] == 'Benchmark':
                        return ['background-color: #ffcccc'] * len(row)
                    else:
                        return [''] * len(row)
                
                styled_dd = formatted_dd.style.apply(highlight_benchmark, axis=1)
                st.dataframe(styled_dd, use_container_width=True, hide_index=True)
                
                # Comparative analysis
                if benchmark_dd_data is not None and len(dd_stats) > 1:
                    st.subheader("🔍 Relative Performance Analysis")
                    
                    benchmark_max_dd = dd_stats[-1]['Max Drawdown']  # Benchmark is last in list
                    
                    better_funds = []
                    worse_funds = []
                    
                    for stat in dd_stats[:-1]:  # Exclude benchmark
                        fund_max_dd = stat['Max Drawdown']
                        if fund_max_dd > benchmark_max_dd:  # Less negative = better
                            better_funds.append(stat['Fund/Benchmark'])
                        else:
                            worse_funds.append(stat['Fund/Benchmark'])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if better_funds:
                            st.success(f"**Funds with Lower Max Drawdown than {benchmark_name}:**")
                            for fund in better_funds:
                                st.write(f"✅ {fund}")
                        else:
                            st.info("No funds outperformed benchmark on max drawdown")
                    
                    with col2:
                        if worse_funds:
                            st.warning(f"**Funds with Higher Max Drawdown than {benchmark_name}:**")
                            for fund in worse_funds:
                                st.write(f"❌ {fund}")
                        else:
                            st.success("All funds outperformed benchmark on max drawdown!")

def show_stress_scenario_analysis(returns_data, mapping_data):
    """Taleb-inspired stress testing focusing on tail risks and non-normal distributions"""
    
    st.subheader("⚡ Tail Risk & Stress Testing (Taleb Framework)")
    
    show_info_button(
        "taleb_overview",
        "About Taleb's Risk Framework",
        """
        **Nassim Taleb's approach to risk management emphasizes:**
        
        • **Black Swan Events**: Rare but impactful events that traditional models miss
        • **Fat Tails**: Real-world distributions have heavier tails than normal distributions
        • **Fragility vs Antifragility**: Some strategies break under stress, others benefit
        • **Empirical over Theoretical**: Use actual data rather than theoretical models
        • **Non-linearity**: Small changes can have disproportionate effects
        
        This module implements these concepts to identify hidden risks in your portfolio.
        """
    )
    
    # Initialize Taleb Risk Analyzer
    risk_analyzer = TalebRiskAnalyzer(returns_data)
    
    # Fund selection
    fund_columns = [col for col in returns_data.columns if col != 'Date']
    
    if len(fund_columns) == 0:
        st.warning("No funds available for analysis")
        return
    
    selected_funds = st.multiselect(
        "Select Funds for Tail Risk Analysis",
        fund_columns,
        default=fund_columns[:min(10, len(fund_columns))],
        key="stress_funds_taleb"
    )
    
    if not selected_funds:
        st.warning("Please select at least one fund")
        return
    
    # Main analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Tail Risk Assessment",
        "🎲 Monte Carlo (Fat Tails)",
        "📉 Historical Worst Cases",
        "⚡ Nonlinearity Detection",
        "🔍 Hidden Risks Analysis"
    ])
    
    with tab1:
        show_tail_risk_assessment(returns_data, selected_funds, risk_analyzer)
    
    with tab2:
        show_fat_tail_monte_carlo(returns_data, selected_funds, risk_analyzer)
    
    with tab3:
        show_historical_worst_cases(returns_data, selected_funds)
    
    with tab4:
        show_nonlinearity_detection(returns_data, selected_funds, risk_analyzer)
    
    with tab5:
        show_hidden_risks_analysis(returns_data, selected_funds, mapping_data, risk_analyzer)

def show_tail_risk_assessment(returns_data, selected_funds, risk_analyzer):
    """Assess tail risks using Taleb's framework"""
    
    st.subheader("📊 Tail Risk Assessment")
    
    show_info_button(
        "tail_risk_info",
        "Understanding Tail Risk Metrics",
        """
        **Key Tail Risk Metrics:**
        
        • **Excess Kurtosis**: Measures "fat tails" - values > 3 indicate fatter tails than normal distribution
        • **Skewness**: Asymmetry of returns - negative skew means more prone to large losses
        • **Left Tail Ratio**: Compares extreme losses (1%) to moderate losses (5%) - higher ratios indicate fatter left tail
        • **Hill Estimator**: Estimates power law behavior in tails - lower values mean heavier tails
        • **Percentile Analysis**: Empirical distribution of returns at various probability levels
        
        **Risk Classification:**
        - 🟢 LOW: Normal-like distribution
        - 🟡 MODERATE: Some fat tail characteristics
        - 🟠 HIGH: Significant tail risks
        - 🔴 EXTREME: Severe tail risks, potential for catastrophic losses
        """
    )
    
    # Calculate tail risk metrics for each fund
    tail_metrics = {}
    
    for fund in selected_funds:
        fund_returns = returns_data[fund].dropna()
        if len(fund_returns) >= 24:
            metrics = risk_analyzer.calculate_tail_metrics(fund_returns)
            tail_metrics[fund] = metrics
    
    if not tail_metrics:
        st.warning("Insufficient data for tail risk analysis")
        return
    
    # 1. Tail Risk Classification
    st.subheader("🎯 Tail Risk Classification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create tail risk score
        for fund, metrics in tail_metrics.items():
            risk_level, color = risk_analyzer.classify_tail_risk(metrics)
            
            st.markdown(f"""
            <div style="background-color: {color}; color: white; padding: 0.5rem; margin: 0.2rem; border-radius: 5px;">
                <strong>{fund}</strong>: {risk_level} Tail Risk
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Taleb's "Turkey Problem" Warning
        st.warning("""
        ⚠️ **The Turkey Problem**: Past absence of extreme events doesn't mean safety.
        Funds with low historical tail risk may be most vulnerable to black swans.
        """)
        
        show_info_button(
            "turkey_problem",
            "The Turkey Problem",
            """
            A turkey is fed for 1000 days, each day confirming its belief that life is safe.
            On day 1001, Thanksgiving arrives. The turkey's confidence was highest precisely
            when the risk was greatest. Similarly, funds with smooth historical returns may
            be accumulating hidden risks that will manifest suddenly.
            """
        )
    
    # 2. Empirical Tail Distribution
    st.subheader("📈 Empirical Tail Analysis")
    
    selected_fund_detail = st.selectbox(
        "Select fund for detailed tail analysis",
        selected_funds,
        key="tail_detail_fund"
    )
    
    if selected_fund_detail:
        fund_returns = returns_data[selected_fund_detail].dropna()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Left tail (losses) analysis
            fig_left = go.Figure()
            
            # Empirical distribution of worst returns
            worst_returns = fund_returns[fund_returns < fund_returns.quantile(0.1)]
            
            fig_left.add_trace(go.Histogram(
                x=worst_returns,
                nbinsx=30,
                name='Empirical',
                histnorm='probability density',
                marker_color='red',
                opacity=0.7
            ))
            
            # Compare with normal distribution
            x_range = np.linspace(worst_returns.min(), worst_returns.quantile(0.1), 100)
            normal_pdf = norm.pdf(x_range, fund_returns.mean(), fund_returns.std())
            
            fig_left.add_trace(go.Scatter(
                x=x_range,
                y=normal_pdf,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='blue', width=2)
            ))
            
            fig_left.update_layout(
                title="Left Tail Distribution (Losses)",
                xaxis_title="Return",
                yaxis_title="Probability Density",
                xaxis=dict(tickformat='.1%'),
                height=400
            )
            
            st.plotly_chart(fig_left, use_container_width=True)
            
            show_info_button(
                "left_tail_chart",
                "Reading This Chart",
                """
                If the red bars (actual data) are higher than the blue line (normal distribution)
                in the far left tail, it means extreme losses are more likely than a normal
                distribution would predict. This is a "fat tail" - the hallmark of fragility.
                """
            )
        
        with col2:
            # Q-Q plot for tail assessment
            fig_qq = go.Figure()
            
            # Calculate theoretical quantiles
            sorted_returns = np.sort(fund_returns)
            n = len(sorted_returns)
            theoretical_quantiles = norm.ppf(np.arange(1, n + 1) / (n + 1))
            
            fig_qq.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=sorted_returns,
                mode='markers',
                name='Data',
                marker=dict(size=5)
            ))
            
            # Add reference line
            min_val = min(theoretical_quantiles.min(), sorted_returns.min())
            max_val = max(theoretical_quantiles.max(), sorted_returns.max())
            fig_qq.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Normal',
                line=dict(color='red', dash='dash')
            ))
            
            fig_qq.update_layout(
                title="Q-Q Plot: Testing for Fat Tails",
                xaxis_title="Theoretical Quantiles (Normal)",
                yaxis_title="Sample Quantiles",
                height=400
            )
            
            st.plotly_chart(fig_qq, use_container_width=True)
            
            show_info_button(
                "qq_plot_info",
                "Understanding Q-Q Plots",
                """
                Points should follow the red diagonal line if returns are normally distributed.
                • **S-shaped curve**: Fat tails on both ends
                • **Points below line on left**: Fat left tail (crash risk)
                • **Points above line on right**: Fat right tail (positive surprises)
                
                Most hedge funds show fat left tails - they're more prone to crashes than booms.
                """
            )
    
    # 3. Tail Risk Summary Table
    st.subheader("📊 Tail Risk Metrics Summary")
    
    # Create DataFrame from metrics
    tail_df = pd.DataFrame(tail_metrics).T
    
    # Format display table
    display_cols = ['excess_kurtosis', 'skewness', 'left_tail_ratio', 'p1', 'min', 'tail_index']
    display_df = tail_df[display_cols].copy()
    
    display_df.columns = ['Excess Kurtosis', 'Skewness', 'Left Tail Ratio', '1% VaR', 'Worst Return', 'Tail Index']
    
    # Format percentages
    display_df['1% VaR'] = display_df['1% VaR'].apply(lambda x: f"{x:.2%}")
    display_df['Worst Return'] = display_df['Worst Return'].apply(lambda x: f"{x:.2%}")
    
    # Format other metrics
    for col in ['Excess Kurtosis', 'Skewness', 'Left Tail Ratio', 'Tail Index']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    st.dataframe(display_df, use_container_width=True)

def show_fat_tail_monte_carlo(returns_data, selected_funds, risk_analyzer):
    """Monte Carlo simulation accounting for fat tails"""
    
    st.subheader("🎲 Fat-Tailed Monte Carlo Simulation")
    
    show_info_button(
        "monte_carlo_info",
        "Fat-Tailed Monte Carlo Explained",
        """
        **Why Fat-Tailed Monte Carlo?**
        
        Traditional Monte Carlo assumes normal distributions, which severely underestimates tail risks.
        Our approach:
        
        1. **Student's t-distribution**: Captures fat tails with degrees of freedom parameter
        2. **Empirical resampling**: For extreme cases, we use actual historical returns
        3. **Tail parameter (df)**: Lower values = fatter tails = higher risk
        
        **Interpretation:**
        - df < 4: Extremely fat tails (infinite kurtosis)
        - df < 10: Significant tail risk
        - df > 30: Approaching normal distribution
        """
    )
    
    # Simulation parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_simulations = st.number_input(
            "Number of Simulations",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000
        )
    
    with col2:
        horizon_months = st.number_input(
            "Time Horizon (months)",
            min_value=1,
            max_value=60,
            value=12,
            step=1
        )
    
    with col3:
        confidence_levels = st.multiselect(
            "Confidence Levels",
            [0.90, 0.95, 0.99, 0.995],
            default=[0.95, 0.99]
        )
    
    if st.button("🚀 Run Fat-Tail Simulation"):
        simulation_results = {}
        
        progress_bar = st.progress(0)
        
        for i, fund in enumerate(selected_funds):
            fund_returns = returns_data[fund].dropna()
            
            if len(fund_returns) < 24:
                continue
            
            progress_bar.progress((i + 1) / len(selected_funds))
            
            # Run simulation using risk analyzer
            results = risk_analyzer.run_fat_tail_monte_carlo(
                fund_returns, 
                n_simulations, 
                horizon_months
            )
            
            if results:
                simulation_results[fund] = results
        
        progress_bar.progress(1.0)
        
        # Display results
        if simulation_results:
            st.subheader("📊 Simulation Results")
            
            # Summary table
            summary_data = []
            for fund, results in simulation_results.items():
                row = {
                    'Fund': fund,
                    'Mean Return': results['mean_return'],
                    'Median Return': results['median_return'],
                    'Tail Parameter (df)': results['tail_parameter']
                }
                
                for conf in confidence_levels:
                    row[f'VaR {int(conf*100)}%'] = results[f'var_{int(conf*100)}']
                    row[f'CVaR {int(conf*100)}%'] = results[f'cvar_{int(conf*100)}']
                
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data).set_index('Fund')
            
            # Add risk warning based on tail parameter
            def get_tail_warning(df):
                if df < 4:
                    return "🔴 EXTREME"
                elif df < 10:
                    return "🟡 HIGH"
                elif df < 30:
                    return "🟢 MODERATE"
                else:
                    return "🟢 LOW"
            
            summary_df['Tail Risk'] = summary_df['Tail Parameter (df)'].apply(get_tail_warning)
            
            # Format for display
            for col in summary_df.columns:
                if 'Return' in col or 'VaR' in col or 'CVaR' in col:
                    summary_df[col] = summary_df[col].apply(lambda x: f"{x:.2%}")
                elif col == 'Tail Parameter (df)':
                    summary_df[col] = summary_df[col].apply(lambda x: f"{x:.1f}")
            
            st.dataframe(summary_df, use_container_width=True)
            
            # Visualization
            selected_fund_viz = st.selectbox(
                "Select fund for visualization",
                list(simulation_results.keys())
            )
            
            if selected_fund_viz:
                results = simulation_results[selected_fund_viz]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sample paths
                    fig_paths = go.Figure()
                    
                    for i, path in enumerate(results['sample_paths'][:50]):
                        fig_paths.add_trace(go.Scatter(
                            y=path - 1,
                            mode='lines',
                            line=dict(width=1, color='lightblue'),
                            showlegend=False,
                            opacity=0.3
                        ))
                    
                    # Add mean path
                    mean_path = np.mean([path for path in results['sample_paths']], axis=0)
                    fig_paths.add_trace(go.Scatter(
                        y=mean_path - 1,
                        mode='lines',
                        line=dict(width=3, color='darkblue'),
                        name='Mean Path'
                    ))
                    
                    fig_paths.update_layout(
                        title=f"Sample Paths - {selected_fund_viz}",
                        xaxis_title="Month",
                        yaxis_title="Cumulative Return",
                        yaxis=dict(tickformat='.1%'),
                        height=400
                    )
                    
                    st.plotly_chart(fig_paths, use_container_width=True)
                
                with col2:
                    # Terminal value distribution
                    fig_dist = go.Figure()
                    
                    fig_dist.add_trace(go.Histogram(
                        x=results['terminal_values'] - 1,
                        nbinsx=50,
                        name='Simulated Returns',
                        histnorm='probability density'
                    ))
                    
                    # Add VaR lines
                    for conf in confidence_levels:
                        var_val = results[f'var_{int(conf*100)}']
                        fig_dist.add_vline(
                            x=var_val,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"VaR {int(conf*100)}%"
                        )
                    
                    fig_dist.update_layout(
                        title=f"Terminal Value Distribution - {selected_fund_viz}",
                        xaxis_title=f"{horizon_months}-Month Return",
                        xaxis=dict(tickformat='.1%'),
                        yaxis_title="Probability Density",
                        height=400
                    )
                    
                    st.plotly_chart(fig_dist, use_container_width=True)
            
            # Risk comparison
            st.subheader("⚠️ Tail Risk Comparison")
            
            show_info_button(
                "risk_comparison",
                "Understanding Risk Tradeoffs",
                """
                This chart shows the classic risk-return tradeoff, but with fat-tail adjusted metrics:
                
                • **X-axis (VaR 99%)**: Worst-case scenario (1% probability)
                • **Y-axis (Mean Return)**: Expected return
                • **Ideal position**: Upper right (high return, low tail risk)
                • **Danger zone**: Lower left (low return, high tail risk)
                
                Remember: In Taleb's framework, avoiding large losses is more important than maximizing gains.
                """
            )
            
            if len(summary_data) > 1:
                # Create scatter plot of mean return vs worst-case scenario
                comparison_df = pd.DataFrame(summary_data).set_index('Fund')
                
                # Convert back to numeric for plotting
                for col in comparison_df.columns:
                    if 'Return' in col or 'VaR' in col or 'CVaR' in col:
                        comparison_df[col] = comparison_df[col].str.rstrip('%').astype(float) / 100
                
                if 'VaR 99%' in comparison_df.columns:
                    fig_risk = px.scatter(
                        comparison_df,
                        x='VaR 99%',
                        y='Mean Return',
                        text=comparison_df.index,
                        title="Risk-Return Tradeoff (Fat-Tail Adjusted)"
                    )
                    
                    fig_risk.update_traces(textposition='top center')
                    fig_risk.update_layout(
                        xaxis_title="99% VaR (Worst 1% Outcome)",
                        yaxis_title="Expected Return",
                        xaxis=dict(tickformat='.1%'),
                        yaxis=dict(tickformat='.1%'),
                        height=500
                    )
                    
                    # Add quadrant lines
                    fig_risk.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_risk.add_vline(x=0, line_dash="dash", line_color="gray")
                    
                    st.plotly_chart(fig_risk, use_container_width=True)

def show_historical_worst_cases(returns_data, selected_funds):
    """Analyze actual historical worst cases"""
    
    st.subheader("📉 Historical Worst-Case Analysis")
    
    show_info_button(
        "worst_case_info",
        "The Importance of Worst Cases",
        """
        **Taleb's Principle**: "The worst that has happened is often not the worst that can happen."
        
        This analysis:
        1. Identifies actual historical worst periods
        2. Analyzes recovery patterns
        3. Applies stress multipliers to imagine worse scenarios
        
        **Key Questions:**
        - What if the 2008 crisis was 50% worse?
        - What if recovery took twice as long?
        - What if multiple bad events happened simultaneously?
        """
    )
    
    # Time window selection
    window_size = st.select_slider(
        "Analysis Window (months)",
        options=[1, 3, 6, 12, 24],
        value=3
    )
    
    worst_cases = {}
    
    for fund in selected_funds:
        fund_returns = returns_data[fund].dropna()
        fund_dates = returns_data.loc[fund_returns.index, 'Date']
        fund_returns.index = fund_dates
        
        if len(fund_returns) < window_size:
            continue
        
        # Calculate rolling window returns
        rolling_returns = fund_returns.rolling(window_size).apply(lambda x: np.prod(1 + x) - 1)
        
        # Find worst periods
        worst_periods = rolling_returns.nsmallest(10)
        
        # Store results
        worst_cases[fund] = {
            'worst_periods': worst_periods,
            'worst_return': worst_periods.min(),
            'worst_date': worst_periods.idxmin(),
            'recovery_analysis': analyze_recovery(fund_returns, worst_periods.idxmin(), window_size)
        }
    
    if not worst_cases:
        st.warning("Insufficient data for analysis")
        return
    
    # Display worst cases
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔴 Worst Historical Periods")
        
        worst_summary = []
        for fund, data in worst_cases.items():
            worst_summary.append({
                'Fund': fund,
                'Worst Return': data['worst_return'],
                'Date': data['worst_date'].strftime('%Y-%m') if pd.notna(data['worst_date']) else 'N/A',
                'Recovery Time': data['recovery_analysis']['recovery_months']
            })
        
        worst_df = pd.DataFrame(worst_summary)
        worst_df['Worst Return'] = worst_df['Worst Return'].apply(lambda x: f"{x:.2%}")
        worst_df['Recovery Time'] = worst_df['Recovery Time'].apply(
            lambda x: f"{x} months" if pd.notna(x) and x != -1 else "Not recovered"
        )
        
        st.dataframe(worst_df, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Stress Multipliers")
        
        st.write("What if the worst case was even worse?")
        
        stress_multiplier = st.slider(
            "Stress Multiplier",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1
        )
        
        show_info_button(
            "stress_multiplier",
            "Using Stress Multipliers",
            """
            Stress multipliers help imagine scenarios worse than history:
            
            • **1.5x**: A "bad" scenario becomes "terrible"
            • **2.0x**: Twice as bad as the worst we've seen
            • **3.0x**: Catastrophic scenario (system-wide failure)
            
            Example: If a fund lost 20% in 2008, a 2x multiplier asks "What if it lost 40%?"
            """
        )
        
        stressed_worst = []
        for fund, data in worst_cases.items():
            original_worst = data['worst_return']
            stressed_return = 1 - (1 - original_worst) * stress_multiplier
            
            stressed_worst.append({
                'Fund': fund,
                'Historical Worst': original_worst,
                'Stressed Worst': stressed_return,
                'Additional Loss': stressed_return - original_worst
            })
        
        stressed_df = pd.DataFrame(stressed_worst)
        
        # Format for display
        for col in ['Historical Worst', 'Stressed Worst', 'Additional Loss']:
            stressed_df[col] = stressed_df[col].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(stressed_df, use_container_width=True)
    
    # Detailed fund analysis
    st.subheader("📊 Detailed Worst-Case Analysis")
    
    selected_fund_detail = st.selectbox(
        "Select fund for detailed analysis",
        selected_funds,
        key="worst_case_detail"
    )
    
    if selected_fund_detail and selected_fund_detail in worst_cases:
        fund_data = worst_cases[selected_fund_detail]
        fund_returns = returns_data[selected_fund_detail].dropna()
        
        # Create detailed visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Cumulative Returns with Worst Periods',
                'Distribution of Rolling Returns',
                'Drawdown Analysis',
                'Recovery Path from Worst Case'
            )
        )
        
        # 1. Cumulative returns with worst periods highlighted
        cum_returns = (1 + fund_returns).cumprod()
        
        fig.add_trace(
            go.Scatter(
                x=cum_returns.index,
                y=cum_returns.values,
                mode='lines',
                name='Cumulative Returns',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Highlight worst periods
        for date in fund_data['worst_periods'].index:
            if pd.notna(date):
                fig.add_vrect(
                    x0=date - pd.DateOffset(months=window_size),
                    x1=date,
                    fillcolor="red",
                    opacity=0.2,
                    line_width=0,
                    row=1, col=1
                )
        
        # 2. Distribution of rolling returns
        rolling_returns = fund_returns.rolling(window_size).apply(lambda x: np.prod(1 + x) - 1).dropna()
        
        fig.add_trace(
            go.Histogram(
                x=rolling_returns.values,
                nbinsx=30,
                name='Rolling Returns',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # 3. Drawdown analysis
        drawdowns = calculate_drawdown_series(fund_returns)
        
        fig.add_trace(
            go.Scatter(
                x=drawdowns.index,
                y=drawdowns.values,
                mode='lines',
                name='Drawdown',
                line=dict(color='red'),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        # 4. Recovery path
        if fund_data['recovery_analysis']['recovery_path'] is not None:
            recovery_path = fund_data['recovery_analysis']['recovery_path']
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(recovery_path))),
                    y=recovery_path,
                    mode='lines+markers',
                    name='Recovery Path',
                    line=dict(color='green')
                ),
                row=2, col=2
            )
            
            fig.add_hline(
                y=1.0,
                line_dash="dash",
                line_color="gray",
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=False)
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Return", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Months", row=2, col=2)
        
        fig.update_yaxes(title_text="Cumulative Value", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        fig.update_yaxes(title_text="Recovery Value", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)

def show_nonlinearity_detection(returns_data, selected_funds, risk_analyzer):
    """Detect nonlinearities and convexity in fund returns"""
    
    st.subheader("⚡ Nonlinearity & Convexity Detection")
    
    show_info_button(
        "nonlinearity_info",
        "Understanding Nonlinearity",
        """
        **Taleb's Key Insight**: Many strategies have hidden nonlinearities - they perform well in normal times
        but blow up in extreme conditions.
        
        **Types of Nonlinearity:**
        
        1. **Negative Convexity** (Dangerous):
           - Gains are limited but losses accelerate
           - Example: Selling options, carry trades
           - "Picking up pennies in front of a steamroller"
        
        2. **Positive Convexity** (Desirable):
           - Losses are limited but gains accelerate
           - Example: Long volatility, trend following
           - Benefits from extreme moves
        
        3. **Asymmetric Beta**:
           - Different sensitivity in up vs down markets
           - High downside beta = participates more in crashes
        """
    )
    
    # Select market proxy
    market_options = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Russell 2000": "^RUT",
        "VIX": "^VIX"
    }
    
    selected_market = st.selectbox(
        "Select Market Index for Nonlinearity Analysis",
        list(market_options.keys())
    )
    
    # Download market data
    try:
        # Get date range
        returns_data_clean = returns_data.copy()
        returns_data_clean['Date'] = pd.to_datetime(returns_data_clean['Date'])
        start_date = returns_data_clean['Date'].min()
        end_date = returns_data_clean['Date'].max()
        
        # Download and process market data
        import yfinance as yf
        market_data = yf.download(
            market_options[selected_market],
            start=start_date - pd.DateOffset(days=10),
            end=end_date + pd.DateOffset(days=10),
            progress=False
        )
        
        if market_data.empty:
            st.error("Could not download market data")
            return
        
        # Convert to monthly returns
        market_prices = market_data['Adj Close'] if 'Adj Close' in market_data.columns else market_data['Close']
        market_monthly = market_prices.resample('M').last()
        market_returns = market_monthly.pct_change().dropna()
        
    except Exception as e:
        st.error(f"Error downloading market data: {e}")
        return
    
    # Analyze nonlinearity for each fund
    nonlinearity_results = {}
    
    for fund in selected_funds:
        fund_returns = returns_data[fund].dropna()
        fund_dates = returns_data.loc[fund_returns.index, 'Date']
        fund_returns.index = fund_dates
        
        # Detect nonlinearity using risk analyzer
        results = risk_analyzer.detect_nonlinearity(fund_returns, market_returns)
        
        if results:
            nonlinearity_results[fund] = results
    
    if not nonlinearity_results:
        st.warning("Insufficient data for nonlinearity analysis")
        return
    
    # Display results
    st.subheader("📊 Nonlinearity Detection Results")
    
    # Summary table
    summary_data = []
    for fund, results in nonlinearity_results.items():
        convexity_type = "Neutral"
        warning = ""
        
        if results['convexity'] > 0.5:
            convexity_type = "🟢 Positive Convexity"
        elif results['convexity'] < -0.5:
            convexity_type = "🔴 Negative Convexity"
            warning = "⚠️ DANGER"
        
        beta_asymmetry = None
        if results['up_beta'] is not None and results['down_beta'] is not None:
            beta_asymmetry = results['down_beta'] - results['up_beta']
        
        summary_data.append({
            'Fund': fund,
            'Linear Beta': results['linear_beta'],
            'Convexity': results['convexity'],
            'Type': convexity_type,
            'Up Beta': results['up_beta'],
            'Down Beta': results['down_beta'],
            'Beta Asymmetry': beta_asymmetry,
            'Warning': warning
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Format for display
    for col in ['Linear Beta', 'Convexity', 'Up Beta', 'Down Beta', 'Beta Asymmetry']:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
            )
    
    st.dataframe(summary_df, use_container_width=True)
    
    # Visualization
    selected_fund_nonlin = st.selectbox(
        "Select fund for detailed nonlinearity visualization",
        selected_funds,
        key="nonlin_viz"
    )
    
    if selected_fund_nonlin and selected_fund_nonlin in nonlinearity_results:
        results = nonlinearity_results[selected_fund_nonlin]
        
        # Need to recalculate for visualization
        fund_returns = returns_data[selected_fund_nonlin].dropna()
        fund_dates = returns_data.loc[fund_returns.index, 'Date']
        fund_returns.index = fund_dates
        
        # Align with market returns
        aligned_data = pd.concat([fund_returns, market_returns], axis=1, join='inner').dropna()
        fund_ret = aligned_data.iloc[:, 0]
        market_ret = aligned_data.iloc[:, 1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot with nonlinear fit
            fig_scatter = go.Figure()
            
            # Add scatter points
            fig_scatter.add_trace(go.Scatter(
                x=market_ret,
                y=fund_ret,
                mode='markers',
                name='Actual',
                marker=dict(
                    size=8,
                    color=market_ret,
                    colorscale='RdBu',
                    showscale=True,
                    colorbar=dict(title="Market Return")
                )
            ))
            
            # Add linear fit
            x_range = np.linspace(market_ret.min(), market_ret.max(), 100)
            y_linear = results['linear_beta'] * x_range
            
            fig_scatter.add_trace(go.Scatter(
                x=x_range,
                y=y_linear,
                mode='lines',
                name='Linear Fit',
                line=dict(color='blue', dash='dash')
            ))
            
            # Add quadratic fit if significant
            if abs(results['convexity']) > 0.1:
                y_nonlinear = results['linear_beta'] * x_range + results['convexity'] * x_range**2
                
                fig_scatter.add_trace(go.Scatter(
                    x=x_range,
                    y=y_nonlinear,
                    mode='lines',
                    name='Nonlinear Fit',
                    line=dict(color='red', width=2)
                ))
            
            fig_scatter.update_layout(
                title=f"Market Response: {selected_fund_nonlin}",
                xaxis_title=f"{selected_market} Return",
                yaxis_title="Fund Return",
                xaxis=dict(tickformat='.1%'),
                yaxis=dict(tickformat='.1%'),
                height=400
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            show_info_button(
                "scatter_interpretation",
                "Reading This Chart",
                """
                • **Points above line**: Fund outperformed given market move
                • **Points below line**: Fund underperformed
                • **Curved fit**: Indicates nonlinear relationship
                • **Upward curve**: Positive convexity (good)
                • **Downward curve**: Negative convexity (dangerous)
                """
            )
        
        with col2:
            # Asymmetry visualization
            if results['up_beta'] is not None and results['down_beta'] is not None:
                fig_asym = go.Figure()
                
                categories = ['Up Market', 'Down Market']
                betas = [results['up_beta'], results['down_beta']]
                colors = ['green', 'red']
                
                fig_asym.add_trace(go.Bar(
                    x=categories,
                    y=betas,
                    marker_color=colors,
                    text=[f"{b:.3f}" for b in betas],
                    textposition='auto'
                ))
                
                fig_asym.update_layout(
                    title="Beta Asymmetry",
                    yaxis_title="Beta",
                    height=400
                )
                
                st.plotly_chart(fig_asym, use_container_width=True)
                
                # Warning for negative convexity
                if results['down_beta'] > results['up_beta'] * 1.2:
                    st.error("""
                    🚨 **WARNING: Negative Convexity Detected**
                    
                    This fund loses more in down markets than it gains in up markets.
                    This is a classic sign of strategies that "pick up pennies in front of a steamroller."
                    
                    **Common culprits:**
                    - Short volatility strategies
                    - Carry trades
                    - Leveraged mean reversion
                    - Selling insurance (options, CDS)
                    """)

def show_hidden_risks_analysis(returns_data, selected_funds, mapping_data, risk_analyzer):
    """Analyze hidden risks using Taleb's framework"""
    
    st.subheader("🔍 Hidden Risks Analysis")
    
    show_info_button(
        "hidden_risks_overview",
        "Taleb's Hidden Risk Framework",
        """
        **Four Types of Hidden Risks:**
        
        1. **Correlation Breakdown**: 
           - Diversification fails when you need it most
           - "All correlations go to 1 in a crisis"
        
        2. **Liquidity Evaporation**:
           - What seems liquid becomes illiquid overnight
           - Exit doors become walls in panics
        
        3. **Strategy Crowding**:
           - Popular strategies fail together
           - "The market can remain irrational longer than you can remain solvent"
        
        4. **Model Risk**:
           - The risk that your risk model itself is wrong
           - "The map is not the territory"
        """
    )
    
    # Analyze various hidden risk factors
    hidden_risks_results = {}
    
    for fund in selected_funds:
        fund_returns = returns_data[fund].dropna()
        
        if len(fund_returns) < 36:
            continue
        
        # Get other fund returns for correlation analysis
        other_fund_returns = []
        for other_fund in selected_funds:
            if other_fund != fund:
                other_returns = returns_data[other_fund].dropna()
                other_fund_returns.append(other_returns)
        
        # Analyze hidden risks using risk analyzer
        risks = risk_analyzer.analyze_hidden_risks(fund_returns, other_fund_returns)
        
        if risks:
            hidden_risks_results[fund] = risks
    
    if not hidden_risks_results:
        st.warning("Insufficient data for hidden risk analysis")
        return
    
    # Display hidden risk dashboard
    st.subheader("🎯 Hidden Risk Dashboard")
    
    # Risk scoring
    risk_scores = []
    
    for fund, risks in hidden_risks_results.items():
        score, flags = risk_analyzer.calculate_risk_score(risks)
        
        risk_level = "🟢 Low" if score <= 1 else "🟡 Medium" if score <= 3 else "🔴 High"
        
        risk_scores.append({
            'Fund': fund,
            'Risk Score': score,
            'Risk Level': risk_level,
            'Red Flags': ', '.join(flags) if flags else 'None'
        })
    
    risk_df = pd.DataFrame(risk_scores)
    st.dataframe(risk_df, use_container_width=True)
    
    # Detailed analysis
    st.subheader("📊 Detailed Hidden Risk Analysis")
    
    selected_fund_hidden = st.selectbox(
        "Select fund for detailed hidden risk analysis",
        selected_funds,
        key="hidden_risk_detail"
    )
    
    if selected_fund_hidden and selected_fund_hidden in hidden_risks_results:
        risks = hidden_risks_results[selected_fund_hidden]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔄 Path Dependency Risk**")
            
            if 'serial_correlation' in risks:
                serial_corr = risks['serial_correlation']
                
                if abs(serial_corr) > 0.3:
                    st.error(f"High serial correlation: {serial_corr:.3f}")
                    st.write("⚠️ Returns are path-dependent - past performance predicts future")
                else:
                    st.success(f"Low serial correlation: {serial_corr:.3f}")
            
            show_info_button(
                "serial_correlation",
                "Serial Correlation Risk",
                """
                High serial correlation means returns are "sticky" - good months follow good months,
                bad months follow bad months. This creates:
                
                • **Momentum risk**: Trends can reverse violently
                • **Drawdown clustering**: Losses come in groups
                • **False confidence**: Smooth returns until they aren't
                """
            )
            
            st.write("**🔗 Correlation Risk**")
            
            if 'corr_instability' in risks:
                st.metric("Correlation Instability", f"{risks['corr_instability']:.3f}")
                
                if 'max_correlation' in risks:
                    st.metric("Max Correlation (Crisis)", f"{risks['max_correlation']:.3f}")
            
            show_info_button(
                "correlation_risk",
                "Correlation Instability",
                """
                Unstable correlations mean your diversification is unreliable.
                High crisis correlation means the fund becomes more correlated
                with other funds during market stress - precisely when you need
                diversification most.
                """
            )
        
        with col2:
            st.write("**📉 Drawdown Clustering**")
            
            if 'max_bad_streak' in risks:
                st.metric("Longest Loss Streak", f"{int(risks['max_bad_streak'])} months")
                
                if 'avg_bad_streak' in risks:
                    st.metric("Average Loss Streak", f"{risks['avg_bad_streak']:.1f} months")
            
            show_info_button(
                "clustering_risk",
                "Loss Clustering",
                """
                When losses cluster together:
                • Recovery becomes harder
                • Investor confidence erodes
                • Redemption pressure builds
                • Manager may take desperate risks
                """
            )
            
            st.write("**💧 Liquidity Risk**")
            
            if 'illiquidity_score' in risks:
                score = risks['illiquidity_score']
                
                if score > 0.1:
                    st.error(f"High illiquidity score: {score:.3f}")
                    st.write("⚠️ Returns show reversal patterns suggesting liquidity issues")
                else:
                    st.success(f"Low illiquidity score: {score:.3f}")
            
            show_info_button(
                "liquidity_risk",
                "Liquidity Risk Indicators",
                """
                Return reversals suggest illiquidity:
                • Large positive returns followed by negatives (and vice versa)
                • Indicates difficulty executing at fair prices
                • Warning sign of crowded trades
                • Risk of being trapped in positions
                """
            )
    
    # Strategy concentration analysis
    if 'Strategy' in mapping_data.columns:
        st.subheader("🎯 Strategy Concentration Risk")
        
        # Merge returns with strategy info
        strategy_mapping = mapping_data.set_index('Fund Name')['Strategy'].to_dict()
        
        # Calculate strategy correlations
        strategy_groups = {}
        for fund in selected_funds:
            if fund in strategy_mapping:
                strategy = strategy_mapping[fund]
                if strategy not in strategy_groups:
                    strategy_groups[strategy] = []
                strategy_groups[strategy].append(fund)
        
        # Show concentration
        strategy_concentration = []
        for strategy, funds in strategy_groups.items():
            if len(funds) > 1:
                # Calculate average pairwise correlation
                correlations = []
                for i in range(len(funds)):
                    for j in range(i+1, len(funds)):
                        fund1_returns = returns_data[funds[i]].dropna()
                        fund2_returns = returns_data[funds[j]].dropna()
                        aligned = pd.concat([fund1_returns, fund2_returns], axis=1, join='inner')
                        
                        if len(aligned) > 24:
                            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                            correlations.append(corr)
                
                if correlations:
                    avg_corr = np.mean(correlations)
                    max_corr = np.max(correlations)
                    
                    strategy_concentration.append({
                        'Strategy': strategy,
                        'Fund Count': len(funds),
                        'Avg Correlation': avg_corr,
                        'Max Correlation': max_corr,
                        'Concentration Risk': '🔴 High' if avg_corr > 0.7 else '🟡 Medium' if avg_corr > 0.5 else '🟢 Low'
                    })
        
        if strategy_concentration:
            conc_df = pd.DataFrame(strategy_concentration)
            
            # Format for display
            conc_df['Avg Correlation'] = conc_df['Avg Correlation'].apply(lambda x: f"{x:.3f}")
            conc_df['Max Correlation'] = conc_df['Max Correlation'].apply(lambda x: f"{x:.3f}")
            
            st.dataframe(conc_df, use_container_width=True)
            
            st.warning("""
            ⚠️ **Strategy Crowding Risk**: High correlations within strategies suggest crowding.
            In a crisis, all funds in a crowded strategy may experience losses simultaneously.
            """)
            
            show_info_button(
                "crowding_risk",
                "The Danger of Crowded Trades",
                """
                **Historical Examples of Crowded Trade Failures:**
                
                • **1998 LTCM**: Everyone was in the same "arbitrage" trades
                • **2007 Quant Crisis**: All quant funds had similar positions
                • **2020 March**: Risk parity strategies all deleveraged together
                
                **Warning Signs:**
                - High correlation within strategy (>0.7)
                - Many funds in same strategy
                - Low volatility followed by spike
                - "This time is different" mentality
                """
            )
    
    # Final warnings and recommendations
    st.subheader("⚠️ Key Warnings & Recommendations")
    
    warnings = []
    
    # Check for specific risk patterns
    high_risk_funds = [r['Fund'] for r in risk_scores if r['Risk Score'] > 3]
    if high_risk_funds:
        warnings.append(f"🔴 High hidden risk detected in: {', '.join(high_risk_funds)}")
    
    # Check for correlation concentration
    if strategy_concentration and any(float(sc['Avg Correlation']) > 0.7 for sc in strategy_concentration):
        warnings.append("🔴 Dangerous strategy concentration detected - reduce exposure to crowded strategies")
    
    # Check for fat tails
    fat_tail_funds = []
    for fund in selected_funds:
        returns = returns_data[fund].dropna()
        if len(returns) > 24:
            kurt = kurtosis(returns)
            if kurt > 5:
                fat_tail_funds.append(fund)
    
    if fat_tail_funds:
        warnings.append(f"🔴 Extreme fat tails in: {', '.join(fat_tail_funds)}")
    
    # Display warnings
    for warning in warnings:
        st.error(warning)
    
    # Taleb's recommendations
    st.info("""
    📚 **Taleb's Risk Management Principles:**
    
    1. **Barbell Strategy**: Combine extremely safe assets with small allocations to high-risk/high-reward
    2. **Respect the Unknown**: What you don't know is more important than what you know
    3. **Avoid Negative Convexity**: Never invest in strategies that lose more than they gain
    4. **Redundancy over Optimization**: Multiple uncorrelated strategies beat one "optimal" strategy
    5. **Skin in the Game**: Prefer managers who invest their own money
    """)
    
    show_info_button(
        "taleb_principles",
        "Applying Taleb's Principles",
        """
        **Practical Implementation:**
        
        **1. Barbell Strategy Example:**
        - 80-90% in ultra-safe assets (T-bills)
        - 10-20% in high-risk, high-reward strategies
        - Never in the "middle" (moderate risk/return)
        
        **2. Respect the Unknown:**
        - Assume your worst drawdown is ahead, not behind
        - Plan for scenarios you haven't seen
        - "Absence of evidence ≠ evidence of absence"
        
        **3. Avoid Negative Convexity:**
        - No strategies that "blow up"
        - No picking up pennies in front of steamrollers
        - If it seems too good to be true, it is
        
        **4. Redundancy:**
        - Multiple strategies that can each fail independently
        - Diversification that works in crisis, not just in calm
        - "Inefficient" by design
        
        **5. Skin in the Game:**
        - Managers eating their own cooking
        - Alignment of interests
        - Real downside for decision makers
        """
    )

def show_reports_export(returns_data, mapping_data):
    """Enhanced reports and export functionality"""
    
    st.subheader("📋 Reports & Export")
    
    # Calculate comprehensive metrics
    fund_metrics = calculate_fund_metrics_batch(returns_data)
    
    if fund_metrics is not None and not fund_metrics.empty:
        # Merge with mapping data
        comprehensive_data = fund_metrics.merge(
            mapping_data.set_index('Fund Name'),
            left_index=True,
            right_index=True,
            how='left'
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Generate Reports")
            
            report_type = st.selectbox(
                "Select Report Type",
                [
                    "Executive Summary",
                    "Detailed Performance Report", 
                    "Risk Analysis Report",
                    "Factor Analysis Report",
                    "Taleb Risk Assessment",
                    "Complete Portfolio Analysis"
                ],
                key="hf_report_type"
            )
            
            if st.button("🎯 Generate Report", key="hf_generate_report"):
                with st.spinner("Generating report..."):
                    try:
                        # Create report data based on type
                        report_data = {}
                        
                        if report_type == "Executive Summary":
                            summary_cols = ['Ann_Return', 'Ann_Vol', 'Sharpe', 'Max_DD', 'Strategy', 'Region']
                            available_cols = [col for col in summary_cols if col in comprehensive_data.columns]
                            report_data["Executive_Summary"] = comprehensive_data[available_cols]
                        
                        elif report_type == "Detailed Performance Report":
                            perf_cols = ['Ann_Return', 'Ann_Vol', 'Sharpe', 'Sortino', 'Calmar', 'Omega']
                            available_cols = [col for col in perf_cols if col in comprehensive_data.columns]
                            report_data["Performance_Metrics"] = comprehensive_data[available_cols]
                        
                        elif report_type == "Risk Analysis Report":
                            risk_cols = ['Ann_Vol', 'Max_DD', 'VaR_95', 'CVaR_95', 'Downside_Risk', 'Skew', 'Kurtosis']
                            available_cols = [col for col in risk_cols if col in comprehensive_data.columns]
                            report_data["Risk_Metrics"] = comprehensive_data[available_cols]
                        
                        elif report_type == "Taleb Risk Assessment":
                            # Run Taleb analysis for export
                            risk_analyzer = TalebRiskAnalyzer(returns_data)
                            taleb_results = []
                            
                            for fund in comprehensive_data.index:
                                if fund in returns_data.columns:
                                    fund_returns = returns_data[fund].dropna()
                                    if len(fund_returns) >= 24:
                                        metrics = risk_analyzer.calculate_tail_metrics(fund_returns)
                                        risk_level, _ = risk_analyzer.classify_tail_risk(metrics)
                                        
                                        taleb_results.append({
                                            'Fund': fund,
                                            'Excess_Kurtosis': metrics.get('excess_kurtosis', np.nan),
                                            'Skewness': metrics.get('skewness', np.nan),
                                            'Left_Tail_Ratio': metrics.get('left_tail_ratio', np.nan),
                                            'Tail_Risk_Level': risk_level
                                        })
                            
                            report_data["Taleb_Risk_Assessment"] = pd.DataFrame(taleb_results)
                        
                        else:  # Complete analysis
                            report_data["Complete_Analysis"] = comprehensive_data
                            report_data["Fund_Mapping"] = mapping_data.set_index('Fund Name')
                        
                        # Generate Excel file
                        buffer = export_data_to_excel(report_data, f"{report_type.replace(' ', '_')}.xlsx")
                        
                        if buffer:
                            st.download_button(
                                label="📥 Download Report",
                                data=buffer,
                                file_name=f"HF_{report_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="hf_report_download"
                            )
                        else:
                            st.error("Failed to generate report")
                            
                    except Exception as e:
                        st.error(f"Error generating report: {e}")
        
        with col2:
            st.subheader("📈 Portfolio Insights")
            
            # Generate insights
            insights = []
            
            try:
                # Top performer
                if 'Sharpe' in fund_metrics.columns:
                    top_sharpe_fund = fund_metrics['Sharpe'].idxmax()
                    top_sharpe_value = fund_metrics['Sharpe'].max()
                    insights.append(f"🏆 Best Sharpe Ratio: {top_sharpe_fund} ({top_sharpe_value:.2f})")
                
                # Strategy insights
                if 'Strategy' in comprehensive_data.columns:
                    strategy_counts = comprehensive_data['Strategy'].value_counts()
                    top_strategy = strategy_counts.index[0]
                    insights.append(f"📊 Most Common Strategy: {top_strategy} ({strategy_counts.iloc[0]} funds)")
                
                # Risk insights
                if 'Max_DD' in fund_metrics.columns:
                    avg_dd = fund_metrics['Max_DD'].mean()
                    insights.append(f"📉 Average Max Drawdown: {avg_dd:.2%}")
                
                # Return insights
                if 'Ann_Return' in fund_metrics.columns:
                    avg_return = fund_metrics['Ann_Return'].mean()
                    positive_returns = (fund_metrics['Ann_Return'] > 0).sum()
                    total_funds = len(fund_metrics)
                    insights.append(f"📈 Average Annual Return: {avg_return:.2%}")
                    insights.append(f"✅ Funds with Positive Returns: {positive_returns}/{total_funds}")
                
                # Tail risk warning
                if 'Kurtosis' in fund_metrics.columns:
                    high_kurtosis_funds = (fund_metrics['Kurtosis'] > 5).sum()
                    if high_kurtosis_funds > 0:
                        insights.append(f"⚠️ Funds with extreme fat tails: {high_kurtosis_funds}")
                
                for insight in insights:
                    st.info(insight)
                    
            except Exception as e:
                st.warning(f"Could not generate insights: {e}")




# ─── ChatGPT API Module ─────────────────────────────────────────────────────
def show_chatgpt_api_module(returns_data=None, mapping_data=None, pe_data=None):
    """ChatGPT API Integration for Portfolio Analysis"""
    
    st.markdown('<h1 class="main-header">🤖 Merrimac AI Portfolio Assistant (ClarkGPT)</h1>', unsafe_allow_html=True)
    
    # API Key Management
    api_key = manage_api_key()
    
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key to use the AI Assistant")
        st.info("""
        💡 **How to get an API key:**
        1. Go to [OpenAI Platform](https://platform.openai.com/)
        2. Sign up or log in
        3. Navigate to API keys section
        4. Create a new API key
        5. Copy and paste it above
        """)
        return
    
    # Initialize OpenAI client
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chat Assistant",
        "📊 Data Analysis",
        "📈 Insights Generator",
        "📋 Report Writer"
    ])
    
    with tab1:
        show_chat_assistant(client, returns_data, mapping_data, pe_data)
    
    with tab2:
        show_data_analysis_assistant(client, returns_data, mapping_data, pe_data)
    
    with tab3:
        show_insights_generator(client, returns_data, mapping_data, pe_data)
    
    with tab4:
        show_report_writer(client, returns_data, mapping_data, pe_data)

def manage_api_key():
    """Manage OpenAI API key securely"""
    
    # Check if API key is already in session state
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""
    
    with st.expander("🔑 API Key Configuration", expanded=not st.session_state.openai_api_key):
        api_key = st.text_input(
            "Enter your OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            help="Your API key is stored only for this session and is not saved permanently"
        )
        
        if api_key:
            st.session_state.openai_api_key = api_key
            st.success("✅ API key configured")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Test Connection"):
                if api_key:
                    try:
                        import openai
                        client = openai.OpenAI(api_key=api_key)
                        # Test the connection
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": "Hello"}],
                            max_tokens=10
                        )
                        st.success("✅ Connection successful!")
                    except Exception as e:
                        st.error(f"❌ Connection failed: {e}")
                else:
                    st.warning("Please enter an API key first")
        
        with col2:
            if st.button("Clear API Key"):
                st.session_state.openai_api_key = ""
                st.success("API key cleared")
    
    return st.session_state.openai_api_key

def show_chat_assistant(client, returns_data, mapping_data, pe_data):
    """Interactive chat assistant for portfolio questions"""
    
    st.subheader("💬 Portfolio Chat Assistant")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Context selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("""
        Ask me anything about your portfolio! I can help with:
        - Performance analysis and comparisons
        - Risk assessment and recommendations
        - Strategy insights and allocations
        - Market outlook and positioning
        - Custom calculations and metrics
        """)
    
    with col2:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared")
    
    # Data context checkboxes
    st.write("**Select data context for the assistant:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_hf_data = st.checkbox("Include Hedge Fund Data", value=bool(returns_data is not None))
    with col2:
        use_pe_data = st.checkbox("Include Private Equity Data", value=bool(pe_data is not None))
    with col3:
        use_mapping_data = st.checkbox("Include Fund Metadata", value=bool(mapping_data is not None))
    
    # Chat interface
    user_input = st.text_area(
        "Ask your question:",
        placeholder="e.g., Which funds have the best risk-adjusted returns? What's my exposure to emerging markets?",
        height=100
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("🚀 Send", type="primary"):
            if user_input:
                with st.spinner("Thinking..."):
                    response = get_chatgpt_response(
                        client, 
                        user_input, 
                        returns_data if use_hf_data else None,
                        mapping_data if use_mapping_data else None,
                        pe_data if use_pe_data else None,
                        st.session_state.chat_history
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
    
    with col2:
        # Check available models based on user's access
        available_models = ["gpt-3.5-turbo"]
        if st.session_state.get('has_gpt4_access', False):
            available_models.extend(["gpt-4", "gpt-4-turbo-preview"])
        
        model = st.selectbox(
            "Model",
            available_models,
            index=0
        )
        st.session_state['selected_model'] = model
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("📜 Conversation History")
        
        for i in range(len(st.session_state.chat_history) - 1, -1, -1):
            msg = st.session_state.chat_history[i]
            
            if msg["role"] == "user":
                st.markdown(f"**🧑 You:** {msg['content']}")
            else:
                st.markdown(f"**🤖 Assistant:** {msg['content']}")
            
            if i > 0:
                st.markdown("---")

def show_data_analysis_assistant(client, returns_data, mapping_data, pe_data):
    """AI-powered data analysis suggestions"""
    
    st.subheader("📊 AI Data Analysis Assistant")
    
    analysis_type = st.selectbox(
        "Select analysis type:",
        [
            "Performance Attribution",
            "Risk Decomposition",
            "Correlation Analysis",
            "Outlier Detection",
            "Trend Analysis",
            "Benchmark Comparison",
            "Custom Analysis"
        ]
    )
    
    if analysis_type == "Custom Analysis":
        custom_request = st.text_area(
            "Describe your analysis request:",
            placeholder="e.g., Analyze the correlation between my equity funds and interest rates over the past 2 years"
        )
    
    if st.button("🔍 Generate Analysis"):
        with st.spinner("Analyzing your portfolio data..."):
            # Prepare data summary for context
            data_summary = prepare_data_summary(returns_data, mapping_data, pe_data)
            
            # Create analysis prompt
            if analysis_type == "Custom Analysis":
                prompt = f"""
                Based on the following portfolio data summary, please provide a detailed {custom_request}:
                
                {data_summary}
                
                Please provide:
                1. Key findings
                2. Statistical insights
                3. Visual recommendations
                4. Action items
                """
            else:
                prompt = f"""
                Based on the following portfolio data summary, please provide a comprehensive {analysis_type}:
                
                {data_summary}
                
                Include:
                1. Detailed analysis results
                2. Key metrics and statistics
                3. Interpretation of findings
                4. Recommendations based on the analysis
                """
            
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Changed from gpt-4
                    messages=[
                        {"role": "system", "content": "You are a expert portfolio analyst specializing in hedge funds and private equity. Provide detailed, quantitative analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.7
                )
                
                analysis_result = response.choices[0].message.content
                
                # Display results
                st.markdown("### 📈 Analysis Results")
                st.markdown(analysis_result)
                
                # Offer to save results
                if st.button("💾 Save Analysis"):
                    save_analysis_to_file(analysis_type, analysis_result)
                    
            except Exception as e:
                st.error(f"Error generating analysis: {e}")

def show_insights_generator(client, returns_data, mapping_data, pe_data):
    """Generate AI-powered insights"""
    
    st.subheader("📈 AI Insights Generator")
    
    insight_categories = st.multiselect(
        "Select insight categories:",
        [
            "Performance Insights",
            "Risk Warnings",
            "Opportunity Identification",
            "Portfolio Optimization",
            "Market Positioning",
            "Peer Comparison",
            "Forward-Looking Analysis"
        ],
        default=["Performance Insights", "Risk Warnings"]
    )
    
    depth = st.slider("Analysis Depth", 1, 5, 3, help="1=High-level, 5=Very detailed")
    
    if st.button("🎯 Generate Insights"):
        with st.spinner("Generating insights..."):
            # Prepare comprehensive data context
            data_context = prepare_comprehensive_context(returns_data, mapping_data, pe_data)
            
            insights = {}
            
            for category in insight_categories:
                prompt = f"""
                As an expert portfolio analyst, generate {depth} key insights for '{category}' based on this portfolio data:
                
                {data_context}
                
                For each insight:
                1. Provide a clear, actionable title
                2. Explain the finding with specific data points
                3. Include implications and recommendations
                4. Assign priority (High/Medium/Low)
                
                Format as a structured list of insights.
                """
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",  # Changed from gpt-4
                        messages=[
                            {"role": "system", "content": "You are a senior portfolio analyst providing actionable insights for a family office."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1500,
                        temperature=0.8
                    )
                    
                    insights[category] = response.choices[0].message.content
                    
                except Exception as e:
                    st.error(f"Error generating {category}: {e}")
            
            # Display insights
            for category, content in insights.items():
                with st.expander(f"📊 {category}", expanded=True):
                    st.markdown(content)
            
            # Summary dashboard
            st.subheader("📋 Insights Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Insights Generated", len(insights))
            with col2:
                st.metric("Categories Analyzed", len(insight_categories))
            with col3:
                st.metric("Analysis Depth", f"{depth}/5")

def show_report_writer(client, returns_data, mapping_data, pe_data):
    """AI-powered report generation"""
    
    st.subheader("📋 AI Report Writer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Report Type:",
            [
                "Executive Summary",
                "Quarterly Performance Report",
                "Risk Assessment Report",
                "Investment Committee Memo",
                "Due Diligence Report",
                "Custom Report"
            ]
        )
    
    with col2:
        report_length = st.select_slider(
            "Report Length:",
            options=["Brief (1 page)", "Standard (2-3 pages)", "Detailed (4-5 pages)", "Comprehensive (6+ pages)"],
            value="Standard (2-3 pages)"
        )
    
    # Additional options
    include_recommendations = st.checkbox("Include Recommendations", value=True)
    include_charts = st.checkbox("Suggest Charts/Visualizations", value=True)
    formal_tone = st.checkbox("Use Formal Tone", value=True)
    
    if report_type == "Custom Report":
        custom_requirements = st.text_area(
            "Describe your report requirements:",
            placeholder="e.g., Monthly update for investment committee focusing on emerging market exposure and currency risks"
        )
    
    if st.button("📝 Generate Report"):
        with st.spinner("Writing your report..."):
            # Prepare comprehensive data for report
            full_context = prepare_full_portfolio_context(returns_data, mapping_data, pe_data)
            
            # Build report prompt
            tone = "formal and professional" if formal_tone else "clear and conversational"
            
            prompt = f"""
            Generate a {report_length} {report_type} with the following requirements:
            
            Tone: {tone}
            Include Recommendations: {include_recommendations}
            Include Chart Suggestions: {include_charts}
            
            Portfolio Data Context:
            {full_context}
            
            {"Additional Requirements: " + custom_requirements if report_type == "Custom Report" else ""}
            
            Structure the report with:
            1. Executive Summary
            2. Key Findings
            3. Detailed Analysis
            4. {"Recommendations" if include_recommendations else ""}
            5. {"Suggested Visualizations" if include_charts else ""}
            6. Conclusion
            
            Use specific data points and percentages where available.
            """
            
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Changed from gpt-4
                    messages=[
                        {"role": "system", "content": "You are a senior investment analyst writing reports for a sophisticated family office. Use precise language and data-driven insights."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.7
                )
                
                report_content = response.choices[0].message.content
                
                # Display report
                st.markdown("### 📄 Generated Report")
                
                # Add download button
                st.download_button(
                    label="📥 Download Report (Markdown)",
                    data=report_content,
                    file_name=f"{report_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
                
                # Display report content
                st.markdown(report_content)
                
                # Generate PDF option (requires additional library)
                if st.button("📑 Convert to PDF"):
                    st.info("PDF generation requires additional setup. The markdown file can be converted to PDF using tools like Pandoc.")
                    
            except Exception as e:
                st.error(f"Error generating report: {e}")

# Helper functions for ChatGPT integration
def get_chatgpt_response(client, question, returns_data, mapping_data, pe_data, chat_history):
    """Get response from ChatGPT with portfolio context"""
    
    # Prepare data context
    context = prepare_data_summary(returns_data, mapping_data, pe_data)
    
    # Build conversation history
    messages = [
        {"role": "system", "content": """You are an expert portfolio analyst for a family office specializing in hedge funds and private equity. 
         You have deep knowledge of:
         - Performance analysis and risk metrics
         - Alternative investments and strategies
         - Portfolio construction and optimization
         - Market analysis and trends
         
         Provide specific, data-driven answers using the portfolio data provided. Be concise but thorough."""},
        {"role": "user", "content": f"Portfolio Data Context:\n{context}"}
    ]
    
    # Add chat history (last 5 exchanges for context)
    for msg in chat_history[-10:]:
        messages.append(msg)
    
    # Add current question
    messages.append({"role": "user", "content": question})
    
    # Use the selected model or default to gpt-3.5-turbo
    model = st.session_state.get('selected_model', 'gpt-3.5-turbo')
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # If GPT-4 fails, fall back to GPT-3.5-turbo
        if "gpt-4" in str(e) and model.startswith("gpt-4"):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.7
                )
                st.warning("Note: Using GPT-3.5-turbo as GPT-4 is not available on your account")
                return response.choices[0].message.content
            except Exception as e2:
                return f"Error: {str(e2)}"
        return f"Error: {str(e)}"

def prepare_data_summary(returns_data, mapping_data, pe_data):
    """Prepare summary statistics for ChatGPT context"""
    
    summary = []
    
    # Hedge fund data summary
    if returns_data is not None:
        fund_metrics = calculate_fund_metrics_batch(returns_data)
        if fund_metrics is not None and not fund_metrics.empty:
            summary.append("HEDGE FUND PORTFOLIO SUMMARY:")
            summary.append(f"- Total Funds: {len(fund_metrics)}")
            summary.append(f"- Average Annual Return: {fund_metrics['Ann_Return'].mean():.2%}")
            summary.append(f"- Average Volatility: {fund_metrics['Ann_Vol'].mean():.2%}")
            summary.append(f"- Average Sharpe Ratio: {fund_metrics['Sharpe'].mean():.2f}")
            summary.append(f"- Average Max Drawdown: {fund_metrics['Max_DD'].mean():.2%}")
            
            # Top performers
            top_funds = fund_metrics.nlargest(3, 'Sharpe')
            summary.append("\nTop 3 Funds by Sharpe Ratio:")
            for fund in top_funds.index:
                summary.append(f"  - {fund}: {top_funds.loc[fund, 'Sharpe']:.2f}")
    
    # Mapping data summary
    if mapping_data is not None:
        summary.append("\nFUND ALLOCATION SUMMARY:")
        if 'Strategy' in mapping_data.columns:
            strategy_counts = mapping_data['Strategy'].value_counts()
            summary.append("Strategies:")
            for strategy, count in strategy_counts.items():
                summary.append(f"  - {strategy}: {count} funds")
        
        if 'Region' in mapping_data.columns:
            region_counts = mapping_data['Region'].value_counts()
            summary.append("Regions:")
            for region, count in region_counts.items():
                summary.append(f"  - {region}: {count} funds")
    
    # Private equity data summary
    if pe_data is not None:
        summary.append("\nPRIVATE EQUITY PORTFOLIO SUMMARY:")
        summary.append(f"- Total PE Funds: {len(pe_data)}")
        if 'IRR_Pct' in pe_data.columns:
            summary.append(f"- Average IRR: {pe_data['IRR_Pct'].mean():.1f}%")
        if 'MOIC' in pe_data.columns:
            summary.append(f"- Average MOIC: {pe_data['MOIC'].mean():.2f}x")
        if 'Vintage_Year' in pe_data.columns:
            summary.append(f"- Vintage Years: {pe_data['Vintage_Year'].min()}-{pe_data['Vintage_Year'].max()}")
    
    return "\n".join(summary)

def prepare_comprehensive_context(returns_data, mapping_data, pe_data):
    """Prepare detailed context for insights generation"""
    
    context = prepare_data_summary(returns_data, mapping_data, pe_data)
    
    # Add more detailed metrics
    if returns_data is not None:
        fund_metrics = calculate_fund_metrics_batch(returns_data)
        if fund_metrics is not None and not fund_metrics.empty:
            context += "\n\nDETAILED METRICS:"
            
            # Risk metrics
            if 'Skew' in fund_metrics.columns:
                context += f"\n- Average Skewness: {fund_metrics['Skew'].mean():.2f}"
                context += f"\n- Funds with Negative Skew: {(fund_metrics['Skew'] < 0).sum()}"
            
            if 'Kurtosis' in fund_metrics.columns:
                context += f"\n- Average Kurtosis: {fund_metrics['Kurtosis'].mean():.2f}"
                context += f"\n- Funds with Fat Tails (Kurt > 3): {(fund_metrics['Kurtosis'] > 3).sum()}"
            
            # Performance persistence
            context += f"\n- Funds with Positive Returns: {(fund_metrics['Ann_Return'] > 0).sum()}/{len(fund_metrics)}"
            
            # Risk-adjusted metrics
            if 'Sortino' in fund_metrics.columns:
                context += f"\n- Average Sortino Ratio: {fund_metrics['Sortino'].mean():.2f}"
            
            if 'Calmar' in fund_metrics.columns:
                context += f"\n- Average Calmar Ratio: {fund_metrics['Calmar'].mean():.2f}"
    
    return context

def prepare_full_portfolio_context(returns_data, mapping_data, pe_data):
    """Prepare complete portfolio context for report generation"""
    
    context = prepare_comprehensive_context(returns_data, mapping_data, pe_data)
    
    # Add time period information
    if returns_data is not None:
        context += f"\n\nTIME PERIOD: {returns_data['Date'].min()} to {returns_data['Date'].max()}"
        context += f"\nTotal Months: {len(returns_data)}"
    
    # Add current allocation details
    if mapping_data is not None and '3/31/2025 MV' in mapping_data.columns:
        total_aum = mapping_data['3/31/2025 MV'].sum()
        context += f"\n\nCURRENT ALLOCATION:"
        context += f"\nTotal AUM: ${total_aum:,.0f}M"
        
        # Strategy allocation
        if 'Strategy' in mapping_data.columns:
            strategy_alloc = mapping_data.groupby('Strategy')['3/31/2025 MV'].sum()
            context += "\n\nStrategy Allocation:"
            for strategy, amount in strategy_alloc.items():
                context += f"\n  - {strategy}: ${amount:,.0f}M ({amount/total_aum*100:.1f}%)"
    
    return context

def save_analysis_to_file(analysis_type, content):
    """Save analysis results to file"""
    filename = f"{analysis_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    st.download_button(
        label="📥 Download Analysis",
        data=content,
        file_name=filename,
        mime="text/plain"
    )




# ─── Main Application ──────────────────────────────────────────────────────
def main():
    """Main application controller with logo and radio button navigation"""
    
    # Display Merrimac Logo and Header
    try:
        # Load and display the Merrimac logo
        from PIL import Image
        logo = Image.open("merrimac_logo.jpg")
        
        # Create header with logo on left and title on right
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.image(logo, width=200)
        
        with col2:
            st.markdown("""
            <div style="display: flex; align-items: center; height: 200px;">
                <h1 style="color: #1f4e79; font-size: 2.8rem; margin: 0; font-weight: 600;">
                    Merrimac Portfolio Analytics Platform
                </h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Add separator line
        st.markdown("""
        <hr style="border: none; height: 3px; background: linear-gradient(90deg, #1f4e79 0%, #4a90e2 100%); margin: 2rem 0;">
        """, unsafe_allow_html=True)
        
    except FileNotFoundError:
        st.error("❌ merrimac_logo.jpg not found. Please ensure the logo file is in the same directory as this script.")
        # Fallback to text-based header
        st.markdown("""
        <div style="text-align: left; margin-bottom: 2rem;">
            <h1 style="color: #1f4e79; font-size: 2.8rem; margin: 0; font-weight: 600;">
                Merrimac Portfolio Analytics Platform
            </h1>
        </div>
        <hr style="border: none; height: 3px; background: linear-gradient(90deg, #1f4e79 0%, #4a90e2 100%); margin: 2rem 0;">
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error loading logo: {e}")
        # Fallback to text-based header
        st.markdown("""
        <div style="text-align: left; margin-bottom: 2rem;">
            <h1 style="color: #1f4e79; font-size: 2.8rem; margin: 0; font-weight: 600;">
                Merrimac Portfolio Analytics Platform
            </h1>
        </div>
        <hr style="border: none; height: 3px; background: linear-gradient(90deg, #1f4e79 0%, #4a90e2 100%); margin: 2rem 0;">
        """, unsafe_allow_html=True)
    
    # Navigation with Radio Buttons - Updated to include API module
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <h3 style="color: #1f4e79; margin-bottom: 1rem;">📊 Select Analysis Module</h3>
        </div>
        """, unsafe_allow_html=True)
        
        analysis_module = st.radio(
            "",
            ["🏦 Hedge Fund Analysis", "🏢 Private Equity Analysis", "🤖 AI Assistant (API)"],
            key="main_navigation",
            horizontal=True,
            label_visibility="collapsed"
        )
    
    # Add minimal spacing before content
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    
    # Initialize data variables
    returns_data = None
    mapping_data = None
    pe_data = None
    
    # Load data if available in session state (from other modules)
    if 'returns_data' in st.session_state:
        returns_data = st.session_state.returns_data
    if 'mapping_data' in st.session_state:
        mapping_data = st.session_state.mapping_data
    if 'pe_data' in st.session_state:
        pe_data = st.session_state.pe_data
    
    # Route to appropriate module
    if analysis_module == "🏦 Hedge Fund Analysis":
        # Store data in session state for API module access
        hf_returns, hf_mapping = show_complete_hedge_fund_analysis()
        if hf_returns is not None:
            st.session_state.returns_data = hf_returns
        if hf_mapping is not None:
            st.session_state.mapping_data = hf_mapping
            
    elif analysis_module == "🏢 Private Equity Analysis":
        # Store PE data in session state for API module access
        pe_result = show_complete_private_equity_page()
        if pe_result is not None:
            st.session_state.pe_data = pe_result
            
    elif analysis_module == "🤖 AI Assistant (API)":
        # Retrieve data from session state
        returns_data = st.session_state.get('returns_data', None)
        mapping_data = st.session_state.get('mapping_data', None)
        pe_data = st.session_state.get('pe_data', None)
        
        show_chatgpt_api_module(returns_data, mapping_data, pe_data)



if __name__ == "__main__":
    main()
