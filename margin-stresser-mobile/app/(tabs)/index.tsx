import React, { useState, useEffect } from 'react';
import { View, ScrollView, Text, TextInput, TouchableOpacity, ActivityIndicator, Dimensions, Alert } from 'react-native';
import { useAppContext } from '../context/AppContext';
import { IconSymbol } from '@/components/ui/icon-symbol';
import { LineChart, BarChart } from 'react-native-gifted-charts';
import { styled } from 'nativewind';

const StyledView = styled(View);
const StyledText = styled(Text);
const StyledTextInput = styled(TextInput);
const StyledTouchableOpacity = styled(TouchableOpacity);
const StyledScrollView = styled(ScrollView);

export default function HomeScreen() {
  const { addToHistory, addScenario, activeScenarioToLoad, markScenarioLoaded, settings } = useAppContext();

  const [tickers, setTickers] = useState<{ symbol: string, percent: string }[]>([
    { symbol: 'AAPL?L=2', percent: '7.5' },
    { symbol: 'MSFT?L=2', percent: '7.5' },
    { symbol: 'AVGO?L=2', percent: '7.5' },
    { symbol: 'AMZN?L=2', percent: '7.5' },
    { symbol: 'META?L=2', percent: '7.5' },
    { symbol: 'NVDA?L=2', percent: '7.5' },
    { symbol: 'GOOGL?L=2', percent: '7.5' },
    { symbol: 'TSLA?L=2', percent: '7.5' },
    { symbol: 'GLD', percent: '20' },
    { symbol: 'VXUS', percent: '15' },
    { symbol: 'TQQQ', percent: '5' },
  ]);
  const [portfolioValue, setPortfolioValue] = useState('10000');
  const [marginDebt, setMarginDebt] = useState('0');
  const [interestRate, setInterestRate] = useState(settings.defaultInterestRate);
  const [maintenanceMargin, setMaintenanceMargin] = useState(settings.defaultMaintenanceMargin);

  // Load Scenario Effect
  useEffect(() => {
    if (activeScenarioToLoad) {
      setTickers(activeScenarioToLoad.tickers);
      setPortfolioValue(activeScenarioToLoad.portfolioValue);
      setMarginDebt(activeScenarioToLoad.marginDebt);
      setInterestRate(activeScenarioToLoad.interestRate);
      setMaintenanceMargin(activeScenarioToLoad.maintenanceMargin);
      markScenarioLoaded(); // Reset flag
    }
  }, [activeScenarioToLoad]);

  const saveCurrentScenario = () => {
    Alert.prompt(
      "Save Scenario",
      "Enter a name for this portfolio scenario:",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Save",
          onPress: (name?: string) => {
            if (name) {
              addScenario({
                name,
                tickers,
                portfolioValue,
                marginDebt,
                interestRate,
                maintenanceMargin
              });
              Alert.alert("Saved", "Scenario saved to 'Saved' tab.");
            }
          }
        }
      ],
      "plain-text"
    );
  };

  const [activeTab, setActiveTab] = useState<'charts' | 'analysis'>('charts');

  // Define result type
  type ResultType = {
    equity: number;
    debt: number;
    leverage: string;
    drawdown: number;
    status: string;
    logs: string[];
    stats?: {
      cagr: number;
      sharpe: number;
      sortino: number;
      max_drawdown: number;
      calmar: number;
      volatility: number;
    };
    monthly_returns?: { year: number; month: number; value: number }[];
    quarterly_returns?: { year: number; quarter: number; value: number }[];
    yearly_returns?: { year: number; value: number }[];
    daily_returns?: { date: string; value: number }[];
    daily_stats?: { best: number; worst: number; positive_pct: number };
    daily_histogram?: { value: number; label: string; frontColor: string }[];
  };

  const [result, setResult] = useState<ResultType | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [subTab, setSubTab] = useState<'annual' | 'quarterly' | 'monthly' | 'daily'>('monthly');

  // Chart Data State
  const [chartData, setChartData] = useState<{ x: Date, equity: number, loan: number, usage: number }[] | null>(null);

  const addTicker = () => {
    if (tickers.length < 25) {
      setTickers([...tickers, { symbol: '', percent: '0' }]);
    }
  };

  const removeTicker = (index: number) => {
    if (tickers.length > 1) {
      const newTickers = [...tickers];
      newTickers.splice(index, 1);
      setTickers(newTickers);
    }
  };

  const updateTicker = (index: number, field: 'symbol' | 'percent', value: string) => {
    const newTickers = [...tickers];
    newTickers[index] = { ...newTickers[index], [field]: value };
    setTickers(newTickers);
  };

  const calculate = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setChartData(null); // Clear chart data on new calculation

    try {
      // 1. Prepare Payload
      // Parse inputs (remove commas etc)
      const startVal = parseFloat(portfolioValue.replace(/,/g, ''));
      const debt = parseFloat(marginDebt.replace(/,/g, ''));
      const rate = parseFloat(interestRate);
      const maint = parseFloat(maintenanceMargin) / 100.0;

      // Construct Ticker Dict
      const tickerDict: Record<string, number> = {};
      let totalPercent = 0;
      tickers.forEach(t => {
        if (t.symbol.trim()) {
          const p = parseFloat(t.percent) || 0;
          tickerDict[t.symbol.toUpperCase()] = p;
          totalPercent += p;
        }
      });

      if (Math.abs(totalPercent - 100) > 0.1) {
        throw new Error(`Total allocation must be 100% (Current: ${totalPercent}%)`);
      }

      const payload = {
        tickers: tickerDict,
        start_val: startVal,
        margin_debt: debt,
        margin_rate: rate,
        maintenance_margin: maint,
        start_date: "2010-01-01", // Default start for now, add DatePicker later
        end_date: new Date().toISOString().split('T')[0], // Today's date
        rebalance_freq: "Quarterly"
      };

      // 2. Fetch from API
      // Use computer's local IP instead of localhost for physical devices/Expo Go
      const API_URL = 'http://192.168.68.55:8000/run_stress_test';

      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`API Error: ${response.status} - ${errText}`);
      }

      const data = await response.json();

      // 3. Process Result
      const finalEquity = data.equity[data.equity.length - 1];
      const finalLoan = data.loan[data.loan.length - 1];
      const maxDrawdown = data.stats?.max_drawdown || 0;

      // Process Chart Data (Downsample to ~50 points for simple SVG rendering)
      const dates = data.dates;
      // Chart Kit can get heavy with too many points, aim for ~30-50
      const step = Math.ceil(dates.length / 40);

      const combinedData = [];

      for (let i = 0; i < dates.length; i += step) {
        const dateObj = new Date(dates[i]);
        combinedData.push({
          x: dateObj,
          equity: data.equity[i],
          loan: data.loan[i],
          usage: data.margin_usage[i] * 100
        });
      }
      // Ensure last point is included
      const lastIdx = dates.length - 1;
      const lastDate = new Date(dates[lastIdx]);
      combinedData.push({
        x: lastDate,
        equity: data.equity[lastIdx],
        loan: data.loan[lastIdx],
        usage: data.margin_usage[lastIdx] * 100
      });

      setChartData(combinedData);

      setResult({
        equity: finalEquity,
        debt: finalLoan,
        leverage: ((finalEquity + finalLoan) / finalEquity).toFixed(2),
        drawdown: maxDrawdown,
        status: "Stress Test Complete",
        logs: data.logs,
        stats: data.stats,
        monthly_returns: data.monthly_returns,
        quarterly_returns: data.quarterly_returns,
        yearly_returns: data.yearly_returns,
        daily_returns: data.daily_returns,
        daily_histogram: data.daily_histogram
      });

      // Add to History
      addToHistory({
        date: new Date().toISOString(),
        scenarioName: "Stress Test Run", // Could prompt for name or use "Custom"
        resultSummary: {
          finalEquity: finalEquity,
          cagr: data.stats?.cagr || 0,
          maxDrawdown: maxDrawdown,
          leverage: ((finalEquity + finalLoan) / finalEquity).toFixed(2)
        }
      });

    } catch (err: any) {
      console.error(err);
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  const screenWidth = Dimensions.get("window").width;

  // Prepare Gifted Charts Data
  const equityData = chartData?.map(d => ({ value: d.equity, label: d.x.getFullYear().toString(), date: d.x.toLocaleDateString() })) || [];
  const loanData = chartData?.map(d => ({ value: d.loan })) || [];
  // For secondary axis (Usage %)
  const usageData = chartData?.map(d => ({ value: d.usage })) || [];

  // New Helper for Color Scale
  const getCellColor = (value: number) => {
    // Simple Green/Red scale
    if (value > 5) return '#16a34a'; // strong green
    if (value > 2) return '#4ade80'; // medium green
    if (value > 0) return '#bbf7d0'; // light green
    if (value === 0) return '#f3f4f6'; // gray
    if (value > -2) return '#fecaca'; // light red
    if (value > -5) return '#f87171'; // medium red
    return '#dc2626'; // strong red
  };

  const getTextColor = (value: number) => {
    if (Math.abs(value) > 2) return 'white';
    return '#374151';
  };

  return (
    <StyledScrollView className="flex-1 bg-gray-100 p-4">
      {/* Title */}
      <StyledText className="text-3xl font-bold text-center mb-6 mt-6 text-gray-800">Margin Stresser</StyledText>

      {/* Inputs Section (Collapsible in future? Keep visible for now) */}
      <StyledView className="bg-white rounded-xl shadow-sm p-5 mb-5 border border-gray-200">
        <StyledText className="text-xl font-semibold mb-4 text-gray-700">Portfolio Allocation</StyledText>
        {tickers.map((t, index) => (
          <StyledView key={index} className="flex-row mb-2 items-center">
            <StyledView className="flex-1 mr-2">
              <StyledTextInput
                value={t.symbol}
                onChangeText={(text) => updateTicker(index, 'symbol', text)}
                className="border border-gray-300 rounded-lg p-3 bg-gray-50 text-gray-800"
                placeholder="Ticker"
                placeholderTextColor="#9ca3af"
                autoCapitalize="characters"
              />
            </StyledView>
            <StyledView className="w-20 mr-2">
              <StyledTextInput
                value={t.percent}
                onChangeText={(text) => updateTicker(index, 'percent', text)}
                keyboardType="numeric"
                className="border border-gray-300 rounded-lg p-3 bg-gray-50 text-gray-800 text-center"
                placeholder="%"
              />
            </StyledView>
            {tickers.length > 1 && (
              <StyledTouchableOpacity onPress={() => removeTicker(index)} className="bg-red-100 p-3 rounded-lg">
                <StyledText className="text-red-500 font-bold">X</StyledText>
              </StyledTouchableOpacity>
            )}
          </StyledView>
        ))}

        <StyledTouchableOpacity
          onPress={addTicker}
          disabled={tickers.length >= 25}
          className={`mb-6 p-2 rounded-lg items-center ${tickers.length >= 25 ? 'bg-gray-100' : 'bg-gray-200'}`}
        >
          <StyledText className={`${tickers.length >= 25 ? 'text-gray-400' : 'text-gray-700'} font-medium`}>
            {tickers.length >= 25 ? 'Max Tickers Reached (25)' : '+ Add Ticker'}
          </StyledText>
        </StyledTouchableOpacity>

        <StyledTouchableOpacity
          onPress={saveCurrentScenario}
          className="mb-6 bg-purple-100 p-3 rounded-lg flex-row justify-center items-center border border-purple-200"
        >
          <IconSymbol size={20} name="star.fill" color="#9333ea" />
          <StyledText className="text-purple-700 font-bold ml-2">Save Scenario</StyledText>
        </StyledTouchableOpacity>

        <StyledText className="text-xl font-semibold mb-4 text-gray-700">Settings</StyledText>
        <StyledView className="mb-4">
          <StyledText className="text-gray-600 mb-1 font-medium">Portfolio Value ($)</StyledText>
          <StyledTextInput value={portfolioValue} onChangeText={setPortfolioValue} keyboardType="numeric" className="border border-gray-300 rounded-lg p-3 bg-gray-50 text-gray-800" />
        </StyledView>
        <StyledView className="mb-4">
          <StyledText className="text-gray-600 mb-1 font-medium">Margin Debt ($)</StyledText>
          <StyledTextInput value={marginDebt} onChangeText={setMarginDebt} keyboardType="numeric" className="border border-gray-300 rounded-lg p-3 bg-gray-50 text-gray-800" />
        </StyledView>
        <StyledView className="mb-4">
          <StyledText className="text-gray-600 mb-1 font-medium">Margin Interest Rate (%)</StyledText>
          <StyledTextInput value={interestRate} onChangeText={setInterestRate} keyboardType="numeric" className="border border-gray-300 rounded-lg p-3 bg-gray-50 text-gray-800" />
        </StyledView>
        <StyledView className="mb-6">
          <StyledText className="text-gray-600 mb-1 font-medium">Maintenance Margin (%)</StyledText>
          <StyledTextInput value={maintenanceMargin} onChangeText={setMaintenanceMargin} keyboardType="numeric" className="border border-gray-300 rounded-lg p-3 bg-gray-50 text-gray-800" />
        </StyledView>

        <StyledTouchableOpacity onPress={calculate} disabled={loading} className={`p-4 rounded-lg items-center ${loading ? 'bg-blue-400' : 'bg-blue-600 active:bg-blue-700'}`}>
          {loading ? <ActivityIndicator color="white" /> : <StyledText className="text-white font-bold text-lg">Run Stress Test</StyledText>}
        </StyledTouchableOpacity>
        {error && <StyledText className="text-red-500 mt-4 text-center">{error}</StyledText>}
      </StyledView>

      {/* Results Section */}
      {
        result && chartData && (
          <StyledView className="bg-white rounded-xl shadow-sm p-4 mb-8 border border-gray-200">
            {/* Tabs */}
            <StyledView className="flex-row mb-6 bg-gray-100 p-1 rounded-lg">
              <StyledTouchableOpacity
                onPress={() => setActiveTab('charts')}
                className={`flex-1 p-2 rounded-md ${activeTab === 'charts' ? 'bg-white shadow-sm' : ''}`}
              >
                <StyledText className={`text-center font-medium ${activeTab === 'charts' ? 'text-blue-600' : 'text-gray-500'}`}>Charts</StyledText>
              </StyledTouchableOpacity>
              <StyledTouchableOpacity
                onPress={() => setActiveTab('analysis')}
                className={`flex-1 p-2 rounded-md ${activeTab === 'analysis' ? 'bg-white shadow-sm' : ''}`}
              >
                <StyledText className={`text-center font-medium ${activeTab === 'analysis' ? 'text-blue-600' : 'text-gray-500'}`}>Analysis</StyledText>
              </StyledTouchableOpacity>
            </StyledView>

            {activeTab === 'charts' ? (
              // CHARTS VIEW
              <StyledView style={{ overflow: 'hidden' }}>
                <StyledText className="text-xl font-semibold mb-4 text-gray-700">Portfolio Performance</StyledText>
                <LineChart
                  data={equityData}
                  data2={loanData}
                  secondaryData={usageData}
                  height={300}
                  width={screenWidth - 80}
                  spacing={(screenWidth - 80) / equityData.length}
                  initialSpacing={0}
                  color1="#16a34a"
                  color2="#dc2626"
                  secondaryLineConfig={{ color: '#f97316', thickness: 2 }}
                  textColor1="#16a34a"
                  textColor2="#dc2626"
                  dataPointsColor1="#16a34a"
                  dataPointsColor2="#dc2626"
                  thickness1={2}
                  thickness2={2}
                  hideDataPoints
                  xAxisLabelTextStyle={{ color: '#9ca3af', fontSize: 10 }}
                  yAxisTextStyle={{ color: '#9ca3af', fontSize: 10 }}
                  yAxisLabelPrefix="$"
                  formatYLabel={(label) => {
                    const val = parseFloat(label);
                    if (val >= 1000000) return (val / 1000000).toFixed(1) + 'M';
                    if (val >= 1000) return (val / 1000).toFixed(0) + 'k';
                    return val.toString();
                  }}
                  secondaryYAxis={{}}
                  pointerConfig={{
                    pointerStripHeight: 160,
                    pointerStripColor: 'lightgray',
                    pointerStripWidth: 2,
                    pointerColor: 'lightgray',
                    radius: 4,
                    pointerLabelWidth: 100,
                    pointerLabelHeight: 120,
                    activatePointersOnLongPress: false,
                    autoAdjustPointerLabelPosition: true,
                    pointerLabelComponent: (items: any) => {
                      return (
                        <View style={{ height: 120, width: 100, backgroundColor: '#282C3F', borderRadius: 4, justifyContent: 'center', paddingLeft: 16 }}>
                          <Text style={{ color: 'lightgray', fontSize: 10, marginBottom: 6 }}>{items[0].date}</Text>
                          <Text style={{ color: 'white', fontWeight: 'bold', fontSize: 10 }}>Equity</Text>
                          <Text style={{ color: 'lightgray', fontSize: 10, marginBottom: 6 }}>${Math.round(items[0].value).toLocaleString()}</Text>
                          {items[1] && (<><Text style={{ color: 'white', fontWeight: 'bold', fontSize: 10 }}>Loan</Text><Text style={{ color: 'lightgray', fontSize: 10, marginBottom: 6 }}>${Math.round(items[1].value).toLocaleString()}</Text></>)}
                          {items[2] && (<><Text style={{ color: 'white', fontWeight: 'bold', fontSize: 10 }}>Usage</Text><Text style={{ color: 'lightgray', fontSize: 10 }}>{items[2].value.toFixed(1)}%</Text></>)}
                        </View>
                      );
                    },
                  }}
                />

                <StyledText className="text-xl font-semibold mb-2 text-gray-700 mt-6">Results Summary</StyledText>
                <StyledView className="bg-gray-50 p-4 rounded-lg mb-4 border border-gray-100">
                  <StyledText className="text-base text-gray-800 mb-1">Final Equity: <StyledText className="font-bold">${result.equity.toLocaleString()}</StyledText></StyledText>
                  <StyledText className="text-base text-gray-800 mb-1">Final Loan: <StyledText className="font-bold">${result.debt.toLocaleString()}</StyledText></StyledText>
                  <StyledText className="text-base text-gray-800 mb-1">Leverage: <StyledText className="font-bold">{result.leverage}x</StyledText></StyledText>
                  <StyledText className="text-base text-gray-800 mb-1">Max Drawdown: <StyledText className="font-bold text-red-600">{result.drawdown.toFixed(2)}%</StyledText></StyledText>
                </StyledView>

                <StyledText className="text-xl font-semibold mb-2 text-gray-700">Logs</StyledText>
                <StyledScrollView className="h-40 bg-gray-100 p-3 rounded-lg border border-gray-200">
                  {result.logs?.map((l: string, i: number) => (
                    <StyledText key={i} className="text-xs text-gray-600 font-mono mb-1">{l}</StyledText>
                  ))}
                </StyledScrollView>
              </StyledView>
            ) : (
              // ANALYSIS VIEW
              <StyledView>
                <StyledText className="text-xl font-semibold mb-4 text-gray-700">Summary Statistics</StyledText>
                <StyledView className="flex-row flex-wrap justify-between mb-6">
                  {[
                    { label: 'CAGR', value: `${result.stats?.cagr?.toFixed(2)}%` },
                    { label: 'Sharpe', value: result.stats?.sharpe?.toFixed(2) },
                    { label: 'Sortino', value: result.stats?.sortino?.toFixed(2) },
                    { label: 'Max Drawdown', value: `${result.stats?.max_drawdown?.toFixed(2)}%`, color: 'text-red-600' },
                    { label: 'Calmar', value: result.stats?.calmar?.toFixed(2) },
                    { label: 'Volatility', value: `${result.stats?.volatility?.toFixed(2)}%` },
                  ].map((stat, i) => (
                    <StyledView key={i} className="w-[48%] bg-gray-50 p-3 rounded-lg mb-3 border border-gray-100">
                      <StyledText className="text-xs text-gray-500 mb-1">{stat.label}</StyledText>
                      <StyledText className={`text-lg font-bold ${stat.color || 'text-gray-800'}`}>{stat.value || 'N/A'}</StyledText>
                    </StyledView>
                  ))}
                </StyledView>

                {/* Sub Tabs */}
                <StyledView className="flex-row mb-4 bg-gray-100 p-1 rounded-lg">
                  {['Annual', 'Quarterly', 'Monthly', 'Daily'].map((tab) => {
                    const tabKey = tab.toLowerCase() as 'annual' | 'quarterly' | 'monthly' | 'daily';
                    const isActive = subTab === tabKey;
                    return (
                      <StyledTouchableOpacity
                        key={tab}
                        onPress={() => setSubTab(tabKey)}
                        className={`flex-1 p-2 rounded-md ${isActive ? 'bg-white shadow-sm' : ''}`}
                      >
                        <StyledText className={`text-center font-medium ${isActive ? 'text-blue-600' : 'text-gray-500'}`}>{tab}</StyledText>
                      </StyledTouchableOpacity>
                    );
                  })}
                </StyledView>

                {/* Views */}
                {subTab === 'annual' && result.yearly_returns && (
                  <StyledView>
                    <StyledText className="text-lg font-semibold mb-2 text-gray-700">Annual Returns</StyledText>
                    {/* Simple Bar List for Annual */}
                    {result.yearly_returns.map((yr, i) => (
                      <StyledView key={i} className="flex-row items-center mb-2">
                        <StyledText className="w-12 text-gray-600 font-medium">{yr.year}</StyledText>
                        <StyledView className="flex-1 h-6 bg-gray-100 rounded overflow-hidden relative">
                          <StyledView
                            className={`h-full absolute ${yr.value >= 0 ? 'bg-green-500' : 'bg-red-500'}`}
                            style={{
                              width: `${Math.min(Math.abs(yr.value) * 2, 100)}%`, // Scale factor
                              left: 0 // Simplification: just bars from left for now, or maybe center?
                            }}
                          />
                        </StyledView>
                        <StyledText className={`w-16 text-right font-bold ${yr.value >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {yr.value > 0 ? '+' : ''}{yr.value.toFixed(2)}%
                        </StyledText>
                      </StyledView>
                    ))}
                  </StyledView>
                )}

                {subTab === 'quarterly' && (
                  <StyledView>
                    <StyledText className="text-lg font-semibold mb-4 text-gray-700">Quarterly Heatmap</StyledText>

                    {/* QUARTERLY HEATMAP GRID - SAME AS BEFORE */}
                    <ScrollView horizontal showsHorizontalScrollIndicator={true}>
                      <View>
                        {/* Header Row (Quarters) */}
                        <View style={{ flexDirection: 'row', marginBottom: 4 }}>
                          <View style={{ width: 50 }} />
                          {['Q1', 'Q2', 'Q3', 'Q4'].map((q) => (
                            <View key={q} style={{ width: 50, alignItems: 'center' }}>
                              <Text style={{ fontSize: 10, fontWeight: 'bold', color: '#6b7280' }}>{q}</Text>
                            </View>
                          ))}
                        </View>

                        {/* Rows (Years) */}
                        {Array.from(new Set(result.quarterly_returns?.map(r => r.year) || [])).sort((a, b) => b - a).map(year => (
                          <View key={year} style={{ flexDirection: 'row', marginBottom: 2 }}>
                            <View style={{ width: 50, justifyContent: 'center' }}>
                              <Text style={{ fontSize: 12, fontWeight: 'bold', color: '#374151' }}>{year}</Text>
                            </View>
                            {[1, 2, 3, 4].map(quarter => {
                              const rec = result.quarterly_returns?.find(r => r.year === year && r.quarter === quarter);
                              const val = rec ? rec.value : null;
                              return (
                                <View
                                  key={`${year}-Q${quarter}`}
                                  style={{
                                    width: 50,
                                    height: 30,
                                    backgroundColor: val !== null ? getCellColor(val) : '#f9fafb',
                                    marginRight: 2,
                                    justifyContent: 'center',
                                    alignItems: 'center',
                                    borderRadius: 4
                                  }}
                                >
                                  {val !== null && (
                                    <Text style={{ fontSize: 9, color: getTextColor(val), fontWeight: '600' }}>
                                      {val.toFixed(1)}
                                    </Text>
                                  )}
                                </View>
                              );
                            })}
                          </View>
                        ))}
                      </View>
                    </ScrollView>

                    {/* Quarterly List */}
                    <StyledText className="text-lg font-semibold mb-2 mt-6 text-gray-700">Return List</StyledText>
                    <StyledView className="border border-gray-200 rounded-lg overflow-hidden">
                      <StyledView className="flex-row bg-gray-100 p-2 border-b border-gray-200">
                        <StyledText className="flex-1 font-bold text-gray-600">Period</StyledText>
                        <StyledText className="flex-1 font-bold text-gray-600 text-right">Return</StyledText>
                      </StyledView>
                      {result.quarterly_returns?.slice().reverse().map((r, i) => (
                        <StyledView key={i} className={`flex-row p-2 ${i !== result.quarterly_returns!.length - 1 ? 'border-b border-gray-100' : ''}`}>
                          <StyledText className="flex-1 text-gray-800">{r.year} Q{r.quarter}</StyledText>
                          <StyledText className={`flex-1 text-right font-medium ${r.value >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {r.value > 0 ? '+' : ''}{r.value.toFixed(2)}%
                          </StyledText>
                        </StyledView>
                      ))}
                    </StyledView>
                  </StyledView>
                )}

                {subTab === 'monthly' && (
                  <StyledView>
                    <StyledText className="text-lg font-semibold mb-4 text-gray-700">Monthly Retrieves Heatmap</StyledText>

                    {/* MONTHLY HEATMAP GRID - SAME AS BEFORE */}
                    <ScrollView horizontal showsHorizontalScrollIndicator={true}>
                      <View>
                        {/* Header Row (Months) */}
                        <View style={{ flexDirection: 'row', marginBottom: 4 }}>
                          <View style={{ width: 50 }} />
                          {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'].map((m) => (
                            <View key={m} style={{ width: 40, alignItems: 'center' }}>
                              <Text style={{ fontSize: 10, fontWeight: 'bold', color: '#6b7280' }}>{m}</Text>
                            </View>
                          ))}
                        </View>

                        {/* Rows (Years) */}
                        {Array.from(new Set(result.monthly_returns?.map(r => r.year) || [])).sort((a, b) => b - a).map(year => (
                          <View key={year} style={{ flexDirection: 'row', marginBottom: 2 }}>
                            <View style={{ width: 50, justifyContent: 'center' }}>
                              <Text style={{ fontSize: 12, fontWeight: 'bold', color: '#374151' }}>{year}</Text>
                            </View>
                            {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map(month => {
                              const rec = result.monthly_returns?.find(r => r.year === year && r.month === month);
                              const val = rec ? rec.value : null;
                              return (
                                <View
                                  key={`${year}-${month}`}
                                  style={{
                                    width: 40,
                                    height: 30,
                                    backgroundColor: val !== null ? getCellColor(val) : '#f9fafb',
                                    marginRight: 2,
                                    justifyContent: 'center',
                                    alignItems: 'center',
                                    borderRadius: 4
                                  }}
                                >
                                  {val !== null && (
                                    <Text style={{ fontSize: 9, color: getTextColor(val), fontWeight: '600' }}>
                                      {val.toFixed(1)}
                                    </Text>
                                  )}
                                </View>
                              );
                            })}
                          </View>
                        ))}
                      </View>
                    </ScrollView>

                    {/* Monthly List */}
                    <StyledText className="text-lg font-semibold mb-2 mt-6 text-gray-700">Return List</StyledText>
                    <StyledScrollView className="h-96 border border-gray-200 rounded-lg">
                      <StyledView className="flex-row bg-gray-100 p-2 border-b border-gray-200">
                        <StyledText className="flex-1 font-bold text-gray-600">Date</StyledText>
                        <StyledText className="flex-1 font-bold text-gray-600 text-right">Return</StyledText>
                      </StyledView>
                      {result.monthly_returns?.slice().reverse().map((r, i) => (
                        <StyledView key={i} className={`flex-row p-2 border-b border-gray-100`}>
                          <StyledText className="flex-1 text-gray-800">{r.year}-{String(r.month).padStart(2, '0')}</StyledText>
                          <StyledText className={`flex-1 text-right font-medium ${r.value >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {r.value > 0 ? '+' : ''}{r.value.toFixed(2)}%
                          </StyledText>
                        </StyledView>
                      ))}
                    </StyledScrollView>
                  </StyledView>
                )}

                {subTab === 'daily' && result.daily_returns && (
                  <StyledView>
                    <StyledText className="text-lg font-semibold mb-2 text-gray-700">Daily Statistics</StyledText>
                    <StyledView className="flex-row justify-between mb-4">
                      <StyledView className="w-[30%] bg-green-50 p-3 rounded-lg border border-green-100">
                        <StyledText className="text-xs text-green-700 font-bold mb-1">Best Day</StyledText>
                        <StyledText className="text-lg font-bold text-green-800">+{result.daily_stats?.best.toFixed(2)}%</StyledText>
                      </StyledView>
                      <StyledView className="w-[30%] bg-red-50 p-3 rounded-lg border border-red-100">
                        <StyledText className="text-xs text-red-700 font-bold mb-1">Worst Day</StyledText>
                        <StyledText className="text-lg font-bold text-red-800">{result.daily_stats?.worst.toFixed(2)}%</StyledText>
                      </StyledView>
                      <StyledView className="w-[30%] bg-blue-50 p-3 rounded-lg border border-blue-100">
                        <StyledText className="text-xs text-blue-700 font-bold mb-1">Pos. Days</StyledText>
                        <StyledText className="text-lg font-bold text-blue-800">{result.daily_stats?.positive_pct.toFixed(1)}%</StyledText>
                      </StyledView>
                    </StyledView>

                    {result.daily_histogram && (
                      <StyledView className="mb-6">
                        <StyledText className="text-lg font-semibold mb-2 text-gray-700">Distribution of Returns</StyledText>
                        <View style={{ overflow: 'hidden' }}>
                          <BarChart
                            data={result.daily_histogram}
                            barWidth={4}
                            spacing={2}
                            roundedTop
                            hideRules
                            xAxisThickness={0}
                            yAxisThickness={0}
                            yAxisTextStyle={{ color: 'gray' }}
                            xAxisLabelTextStyle={{ color: 'gray', fontSize: 10 }}
                            noOfSections={4}
                            height={150}
                            width={screenWidth - 60}
                          />
                        </View>
                      </StyledView>
                    )}

                    <StyledText className="text-lg font-semibold mb-2 text-gray-700">Daily Returns (List)</StyledText>
                    <StyledScrollView className="h-96 border border-gray-200 rounded-lg">
                      <StyledView className="flex-row bg-gray-100 p-2 border-b border-gray-200">
                        <StyledText className="flex-1 font-bold text-gray-600">Date</StyledText>
                        <StyledText className="flex-1 font-bold text-gray-600 text-right">Return</StyledText>
                      </StyledView>
                      {result.daily_returns.slice().reverse().map((r, i) => (
                        <StyledView key={i} className={`flex-row p-2 border-b border-gray-100`}>
                          <StyledText className="flex-1 text-gray-800">{r.date}</StyledText>
                          <StyledText className={`flex-1 text-right font-medium ${r.value >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {r.value > 0 ? '+' : ''}{r.value.toFixed(2)}%
                          </StyledText>
                        </StyledView>
                      ))}
                    </StyledScrollView>
                  </StyledView>
                )}


              </StyledView>
            )}
          </StyledView>
        )
      }
    </StyledScrollView >
  );
}
