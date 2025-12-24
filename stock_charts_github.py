import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, date
import pytz
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import io
import os
import pandas as pd

# Configuration - Using environment variables for security
TICKERS = ['XBI', 'HOOD', 'UBER', 'CIBR', 'NLR', 'RKLB', 'COST', 'QQQM', 'SPLG', 'HUMN', 'KOID', 'GOOGL', 'NVDA', 'RSP', 'PLTR']

# Get email configuration from environment variables (set as GitHub Secrets)
EMAIL_CONFIG = {
    'sender_email': os.environ.get('EMAIL_SENDER'),
    'sender_password': os.environ.get('EMAIL_PASSWORD'),
    'recipient_email': os.environ.get('EMAIL_RECIPIENT'),
    'smtp_server': os.environ.get('SMTP_SERVER', 'smtp.gmail.com'),  # Default to Gmail
    'smtp_port': int(os.environ.get('SMTP_PORT', '587'))  # Default to 587
}

import exchange_calendars as xcals

def is_market_day():
    """Check if today is a NYSE trading day"""
    nyse = xcals.get_calendar("XNYS")
    today = pd.Timestamp.now().normalize()
    return nyse.is_session(today)

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_moving_averages(df, periods=[50, 200]):
    """Calculate moving averages for the given periods"""
    for period in periods:
        df[f'MA{period}'] = df['Close'].rolling(window=period).mean()
    return df

def create_stock_chart(ticker, months=6, save_to_file=False):
    """Create a candlestick chart with moving averages for a single stock"""
    try:
        # Calculate date range
        eastern = pytz.timezone('America/New_York')
        end_date = datetime.now(eastern)
        # Get extra data to ensure we have enough for 200-day MA
        start_date = end_date - timedelta(days=months*30 + 300)
        
        # Download stock data
        print(f"Downloading data for {ticker}...")
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty:
            print(f"No data available for {ticker}")
            return None
        
        # Clean the data - ensure all values are numeric
        df = df.dropna()  # Remove any rows with missing values
        
        # Calculate moving averages
        df = calculate_moving_averages(df, [50, 200])
        df['RSI'] = calculate_rsi(df)
        
        # Filter to last 6 months for display
        display_start = end_date - timedelta(days=months*30)
        # Convert to timezone-naive for comparison
        df.index = df.index.tz_localize(None)
        df_display = df[df.index >= pd.Timestamp(display_start).tz_localize(None)].copy()
        
        if len(df_display) == 0:
            print(f"No recent data available for {ticker}")
            return None
        
        # Create the chart
        fig, (ax, ax_rsi) = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[3, 1], sharex=True)
        
        # Plot candlesticks with better error handling
        for idx, (date, row) in enumerate(df_display.iterrows()):
            try:
                # Ensure all values are float
                open_price = float(row['Open'])
                close_price = float(row['Close'])
                high_price = float(row['High'])
                low_price = float(row['Low'])
                
                color = 'g' if close_price >= open_price else 'r'
                
                # Plot high-low line
                ax.plot([idx, idx], [low_price, high_price], color='black', linewidth=0.5)
                
                # Plot open-close rectangle
                height = abs(close_price - open_price)
                bottom = min(close_price, open_price)
                if height > 0:  # Only plot if there's a difference
                    ax.bar(idx, height, bottom=bottom, color=color, width=0.6, edgecolor='black', linewidth=0.5)
                else:  # If open == close, plot a thin line
                    ax.plot([idx-0.3, idx+0.3], [close_price, close_price], color='black', linewidth=1)
                    
            except (TypeError, ValueError) as e:
                print(f"Skipping data point at index {idx} for {ticker}: {e}")
                continue
        
        # Plot moving averages with error handling
        x_range = range(len(df_display))
        
        # 50-day MA
        try:
            ma50_values = df_display['MA50'].dropna().values
            if len(ma50_values) > 0:
                ma50_x = [i for i, val in enumerate(df_display['MA50']) if not pd.isna(val)]
                ax.plot(ma50_x, ma50_values, label='50-Day MA', color='blue', linewidth=1.5, alpha=0.7)
        except Exception as e:
            print(f"Could not plot 50-day MA for {ticker}: {e}")
        
        # 200-day MA
        try:
            ma200_values = df_display['MA200'].dropna().values
            if len(ma200_values) > 0:
                ma200_x = [i for i, val in enumerate(df_display['MA200']) if not pd.isna(val)]
                ax.plot(ma200_x, ma200_values, label='200-Day MA', color='red', linewidth=1.5, alpha=0.7)
        except Exception as e:
            print(f"Could not plot 200-day MA for {ticker}: {e}")

        # Plot RSI
        rsi_values = df_display['RSI'].values
        ax_rsi.plot(range(len(df_display)), rsi_values, color='purple', linewidth=1.2)
        ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
        ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.5, linewidth=0.8)
        ax_rsi.fill_between(range(len(df_display)), 30, 70, alpha=0.1, color='gray')
        ax_rsi.set_ylabel('RSI')
        ax_rsi.set_ylim(0, 100)
        ax_rsi.yaxis.tick_right()
        current_rsi = df_display['RSI'].iloc[-1]
        ax_rsi.text(0.02, 0.95, f'RSI: {current_rsi:.1f}', 
            transform=ax_rsi.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_rsi.grid(True, alpha=0.3)
        
        # Format x-axis with better spacing
        num_ticks = min(6, len(df_display))
        tick_spacing = max(1, len(df_display) // num_ticks)
        ax_rsi.set_xticks(range(0, len(df_display), tick_spacing))
        ax.set_xticklabels([df_display.index[i].strftime('%b %y') 
                            for i in range(0, len(df_display), tick_spacing)], 
                           rotation=45, ha='right')
        
        # Labels and title
        ax_rsi.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.set_title(f'{ticker} - Daily Chart with Moving Averages\n'
                f'Last Updated: {end_date.strftime("%Y-%m-%d %I:%M %p")}', 
                fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(0.2, 1.0))
        ax.grid(True, alpha=0.3)
        
        # Add current price annotation with error handling
        try:
            current_price = float(df_display['Close'].iloc[-1])
            if len(df_display) > 1:
                previous_close = float(df_display['Close'].iloc[-2])
                change = current_price - previous_close
                change_pct = (change / previous_close) * 100 if previous_close != 0 else 0
            else:
                change = 0
                change_pct = 0
            
            ax.text(0.02, 0.98, f'Current: ${current_price:.2f}\n'
                               f'Change: {change:+.2f} ({change_pct:+.2f}%)', 
                    transform=ax.transAxes, fontsize=10, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        except Exception as e:
            print(f"Could not add price annotation for {ticker}: {e}")
        
        plt.tight_layout()
        
        # Optionally save to file (for GitHub Actions artifacts)
        if save_to_file:
            os.makedirs('charts', exist_ok=True)
            plt.savefig(f'charts/{ticker}_chart.png', dpi=100, bbox_inches='tight')
            print(f"Saved chart for {ticker} to file")
        
        # Save to bytes buffer for email
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        print(f"✅ Successfully created chart for {ticker}")
        return buf
        
    except Exception as e:
        print(f"Error creating chart for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

import base64

def send_email_with_charts(charts_data):
    """Send email with all chart images embedded for Outlook compatibility"""
    # Verify email configuration
    if not all([EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'], EMAIL_CONFIG['recipient_email']]):
        print("Error: Email configuration missing. Please set GitHub Secrets.")
        return False
    
    msg = MIMEMultipart('alternative')  # Changed to 'alternative' for better compatibility
    msg['Subject'] = f'Stock Portfolio Charts - {datetime.now().strftime("%Y-%m-%d")}'
    msg['From'] = EMAIL_CONFIG['sender_email']
    msg['To'] = EMAIL_CONFIG['recipient_email']
    
    # Create both plain text and HTML versions
    text_body = f"""
    Daily Stock Portfolio Update - {datetime.now().strftime("%B %d, %Y")}
    
    Your stock charts have been generated for: {', '.join([t for t, _ in charts_data])}
    
    Each chart shows daily candlesticks with 50-day (blue) and 200-day (red) moving averages.
    
    Note: If images don't display, you may need to click "Download pictures" in Outlook.
    """
    
    # Create HTML body with base64 embedded images (works better with Outlook)
    html_body = f"""
    <html>
      <head></head>
      <body style="font-family: Arial, sans-serif;">
        <h2 style="color: #333;">Daily Stock Portfolio Update</h2>
        <p>Here are your stock charts for {datetime.now().strftime("%B %d, %Y")}:</p>
        <p style="color: #666;">Each chart shows daily candlesticks with 50-day (blue) and 200-day (red) moving averages.</p>
        <br>
    """
    
    # Add each chart to the email using base64 encoding
    for ticker, chart_buffer in charts_data:
        if chart_buffer:
            # Convert image to base64
            chart_buffer.seek(0)
            encoded = base64.b64encode(chart_buffer.read()).decode()
            
            html_body += f"""
            <div style="margin-bottom: 40px; page-break-inside: avoid;">
                <h3 style="color: #333; margin-bottom: 10px;">{ticker}</h3>
                <img src="data:image/png;base64,{encoded}" 
                     alt="{ticker} Chart" 
                     style="max-width: 100%; width: 800px; height: auto; display: block; border: 1px solid #ddd;">
            </div>
            """
    
    html_body += """
        <br>
        <hr style="border: none; border-top: 1px solid #ddd;">
        <p style="color: gray; font-size: 12px;">
        This is an automated message generated by GitHub Actions. Charts show the last 6 months of daily data.<br>
        If images don't display, you may need to click "Download pictures" or check your email security settings.
        </p>
      </body>
    </html>
    """
    
    # Attach both versions
    msg.attach(MIMEText(text_body, 'plain'))
    msg.attach(MIMEText(html_body, 'html'))
    
    # Send the email
    try:
        print(f"Connecting to {EMAIL_CONFIG['smtp_server']}:{EMAIL_CONFIG['smtp_port']}...")
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            server.send_message(msg)
        print(f"✅ Email sent successfully to {EMAIL_CONFIG['recipient_email']} at {datetime.now()}")
        return True
    except Exception as e:
        print(f"❌ Error sending email: {str(e)}")
        return False
        
def generate_and_send_charts():
    """Main function to generate all charts and send email"""
    print(f"\n🚀 Starting stock chart generation at {datetime.now()}...")
    print(f"Processing {len(TICKERS)} stocks: {', '.join(TICKERS)}")
    
    charts_data = []
    successful = []
    failed = []
    
    for ticker in TICKERS:
        print(f"\n📊 Processing {ticker}...")
        chart_buffer = create_stock_chart(ticker, save_to_file=True)
        if chart_buffer:
            charts_data.append((ticker, chart_buffer))
            successful.append(ticker)
            print(f"✅ {ticker} chart created successfully")
        else:
            failed.append(ticker)
            print(f"❌ {ticker} chart failed")
    
    print(f"\n📈 Summary: {len(successful)} successful, {len(failed)} failed")
    if successful:
        print(f"Successful: {', '.join(successful)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    
    if charts_data:
        print(f"\n📧 Sending email with {len(charts_data)} charts...")
        if send_email_with_charts(charts_data):
            print("\n✅ Job completed successfully!")
            return 0
        else:
            print("\n❌ Job completed with email errors")
            return 1
    else:
        print("\n❌ No charts were generated successfully.")
        return 1

if __name__ == "__main__":
    exit_code = generate_and_send_charts()
    exit(exit_code)
