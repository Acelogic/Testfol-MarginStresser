import React, { useState } from 'react';
import { View, Text, ScrollView, TouchableOpacity, SafeAreaView } from 'react-native';
import { styled } from 'nativewind';

const StyledView = styled(View);
const StyledText = styled(Text);
const StyledScrollView = styled(ScrollView);
const StyledTouchableOpacity = styled(TouchableOpacity);

export default function GuideScreen() {
    const [activeTab, setActiveTab] = useState<'guide' | 'methodology' | 'faq'>('guide');

    return (
        <SafeAreaView className="flex-1 bg-gray-100">
            <StyledView className="flex-1 p-4">
                {/* Header */}
                <StyledText className="text-3xl font-bold text-center mb-6 mt-2 text-gray-800">Documentation</StyledText>

                {/* Tab Switcher */}
                <StyledView className="flex-row mb-6 bg-white p-1 rounded-lg border border-gray-200">
                    <StyledTouchableOpacity
                        onPress={() => setActiveTab('guide')}
                        className={`flex-1 p-2 rounded-md ${activeTab === 'guide' ? 'bg-blue-50 border border-blue-100' : ''}`}
                    >
                        <StyledText className={`text-center font-medium ${activeTab === 'guide' ? 'text-blue-600' : 'text-gray-500'}`}>User Guide</StyledText>
                    </StyledTouchableOpacity>
                    <StyledTouchableOpacity
                        onPress={() => setActiveTab('methodology')}
                        className={`flex-1 p-2 rounded-md ${activeTab === 'methodology' ? 'bg-blue-50 border border-blue-100' : ''}`}
                    >
                        <StyledText className={`text-center font-medium ${activeTab === 'methodology' ? 'text-blue-600' : 'text-gray-500'}`}>Methodology</StyledText>
                    </StyledTouchableOpacity>
                    <StyledTouchableOpacity
                        onPress={() => setActiveTab('faq')}
                        className={`flex-1 p-2 rounded-md ${activeTab === 'faq' ? 'bg-blue-50 border border-blue-100' : ''}`}
                    >
                        <StyledText className={`text-center font-medium ${activeTab === 'faq' ? 'text-blue-600' : 'text-gray-500'}`}>FAQ</StyledText>
                    </StyledTouchableOpacity>
                </StyledView>

                {/* Content Area */}
                <StyledScrollView className="flex-1 bg-white rounded-xl shadow-sm border border-gray-200 p-4" showsVerticalScrollIndicator={false}>

                    {activeTab === 'guide' && (
                        <StyledView className="pb-8">
                            <StyledText className="text-2xl font-bold text-gray-800 mb-4">User Guide</StyledText>
                            <StyledText className="text-gray-600 mb-4 leading-6">
                                Welcome to the Testfol Margin Stresser User Guide. This application allows you to simulate leveraged portfolio performance over historical periods, tracking margin debt, equity levels, and potential margin calls.
                            </StyledText>

                            <StyledText className="text-xl font-semibold text-gray-700 mb-2">1. Configure Global Parameters</StyledText>
                            <StyledText className="text-gray-600 mb-2 leading-6">
                                The main view contains all configuration options:
                            </StyledText>
                            <StyledView className="pl-2 mb-4">
                                <StyledText className="text-gray-600 mb-1">• <StyledText className="font-bold">Portfolio Value</StyledText>: Initial investment.</StyledText>
                                <StyledText className="text-gray-600 mb-1">• <StyledText className="font-bold">Margin Debt</StyledText>: Initial borrowed amount.</StyledText>
                                <StyledText className="text-gray-600 mb-1">• <StyledText className="font-bold">Date Range</StyledText>: Currently fixed (2010-Present).</StyledText>
                            </StyledView>

                            <StyledText className="text-xl font-semibold text-gray-700 mb-2">2. Define Portfolio Allocation</StyledText>
                            <StyledText className="text-gray-600 mb-2 leading-6">
                                Add tickers and their weights. Ensure the total Sum is 100%.
                            </StyledText>
                            <StyledText className="text-gray-600 mb-4 italic">
                                Tip: You can use Testfol modifiers like `SPY?L=2` for 2x leverage.
                            </StyledText>

                            <StyledText className="text-xl font-semibold text-gray-700 mb-2">3. Visualization Modes</StyledText>
                            <StyledText className="text-gray-600 mb-2 leading-6">
                                <StyledText className="font-bold">Charts Tab</StyledText>: Interactive line chart showing Equity, Loan, and Margin Usage.
                            </StyledText>
                            <StyledText className="text-gray-600 mb-4 leading-6">
                                <StyledText className="font-bold">Analysis Tab</StyledText>: Detailed breakdown including Heatmaps (Monthly/Quarterly/Annual) and Return Lists.
                            </StyledText>
                        </StyledView>
                    )}

                    {activeTab === 'methodology' && (
                        <StyledView className="pb-8">
                            <StyledText className="text-2xl font-bold text-gray-800 mb-4">Methodology</StyledText>

                            <StyledText className="text-xl font-semibold text-gray-700 mb-2">Data Source</StyledText>
                            <StyledText className="text-gray-600 mb-4 leading-6">
                                The application is powered by <StyledText className="text-blue-600">testfol.io</StyledText>. Historical price data covers stocks and ETFs back to 1885.
                            </StyledText>

                            <StyledText className="text-xl font-semibold text-gray-700 mb-2">Margin Simulation Logic</StyledText>
                            <StyledText className="text-gray-600 mb-2 leading-6">
                                The margin simulation is applied *on top* of the unleveraged portfolio performance.
                            </StyledText>

                            <StyledText className="font-bold text-gray-700 mt-2 mb-1">Daily Interest Calculation</StyledText>
                            <StyledText className="text-gray-600 mb-2 leading-6">
                                Interest is compounded daily based on the annual interest rate provided:{"\n"}
                                Daily Rate = (Annual Rate / 100) / 252
                            </StyledText>

                            <StyledText className="font-bold text-gray-700 mt-2 mb-1">Margin Call Detection</StyledText>
                            <StyledText className="text-gray-600 mb-4 leading-6">
                                A margin call is triggered when Margin Usage reaches 100%.{"\n"}
                                Usage % = Loan / (Portfolio Value × (1 - Maint %))
                            </StyledText>
                        </StyledView>
                    )}

                    {activeTab === 'faq' && (
                        <StyledView className="pb-8">
                            <StyledText className="text-2xl font-bold text-gray-800 mb-4">FAQ</StyledText>

                            <StyledText className="text-lg font-semibold text-gray-700 mb-1">"Network Request Failed"</StyledText>
                            <StyledText className="text-gray-600 mb-4 leading-6">
                                Ensure your phone/simulator is on the same WiFi as your computer, and the API is running (`python api.py`). The app should be pointing to your local IP, not `localhost`.
                            </StyledText>

                            <StyledText className="text-lg font-semibold text-gray-700 mb-1">Can I simulate short selling?</StyledText>
                            <StyledText className="text-gray-600 mb-4 leading-6">
                                No, the current version only supports long positions with margin leverage.
                            </StyledText>

                            <StyledText className="text-lg font-semibold text-gray-700 mb-1">What does "Maintenance %" mean?</StyledText>
                            <StyledText className="text-gray-600 mb-4 leading-6">
                                It is the minimum amount of equity (as a % of total market value) you must hold.
                                {"\n"}• 25%: Standard Reg T
                                {"\n"}• 30-50%: Volatile stocks
                                {"\n"}• 100%: Cash account
                            </StyledText>
                        </StyledView>
                    )}

                </StyledScrollView>
            </StyledView>
        </SafeAreaView>
    );
}
