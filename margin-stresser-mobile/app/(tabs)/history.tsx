import React from 'react';
import { View, Text, TouchableOpacity, ScrollView, SafeAreaView, Alert } from 'react-native';
import { styled } from 'nativewind';
import { useAppContext } from '../context/AppContext';
import { IconSymbol } from '@/components/ui/icon-symbol';

const StyledView = styled(View);
const StyledText = styled(Text);
const StyledScrollView = styled(ScrollView);
const StyledTouchableOpacity = styled(TouchableOpacity);

export default function HistoryScreen() {
    const { history, clearHistory } = useAppContext();

    const handleClear = () => {
        Alert.alert(
            "Clear History",
            "Are you sure you want to delete all history?",
            [
                { text: "Cancel", style: "cancel" },
                { text: "Clear All", style: "destructive", onPress: clearHistory }
            ]
        );
    };

    return (
        <SafeAreaView className="flex-1 bg-gray-100">
            <StyledView className="flex-1 p-4">
                <StyledView className="flex-row justify-between items-center mb-6 mt-2">
                    <StyledView style={{ width: 24 }} />
                    <StyledText className="text-3xl font-bold text-gray-800">Run History</StyledText>
                    {history.length > 0 ? (
                        <StyledTouchableOpacity onPress={handleClear}>
                            <IconSymbol size={24} name="trash" color="#ef4444" />
                        </StyledTouchableOpacity>
                    ) : <StyledView style={{ width: 24 }} />}
                </StyledView>

                {history.length === 0 ? (
                    <StyledView className="flex-1 justify-center items-center">
                        <StyledText className="text-gray-500 text-lg mb-2">No history yet.</StyledText>
                        <StyledText className="text-gray-400 text-center px-6">
                            Run a stress test in the Home tab to see it appear here.
                        </StyledText>
                    </StyledView>
                ) : (
                    <StyledScrollView className="flex-1" showsVerticalScrollIndicator={false}>
                        {history.map((item) => (
                            <StyledView key={item.id} className="bg-white rounded-xl shadow-sm p-4 mb-3 border border-gray-200">
                                <StyledView className="flex-row justify-between items-center mb-2">
                                    <StyledText className="text-sm font-medium text-gray-500">{new Date(item.date).toLocaleString()}</StyledText>
                                    <StyledText className="text-xs font-bold bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full">{item.scenarioName}</StyledText>
                                </StyledView>

                                <StyledView className="flex-row justify-between mt-1">
                                    <StyledView>
                                        <StyledText className="text-gray-600 text-xs">Final Equity</StyledText>
                                        <StyledText className="text-lg font-bold text-gray-800">${item.resultSummary.finalEquity.toLocaleString()}</StyledText>
                                    </StyledView>
                                    <StyledView className="items-end">
                                        <StyledText className="text-gray-600 text-xs">CAGR</StyledText>
                                        <StyledText className={`text-lg font-bold ${item.resultSummary.cagr >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                            {item.resultSummary.cagr.toFixed(2)}%
                                        </StyledText>
                                    </StyledView>
                                </StyledView>

                                <StyledView className="flex-row justify-between mt-3 border-t border-gray-100 pt-2">
                                    <StyledText className="text-xs text-gray-500">Max DD: <StyledText className="text-red-600 font-bold">{item.resultSummary.maxDrawdown.toFixed(2)}%</StyledText></StyledText>
                                    <StyledText className="text-xs text-gray-500">Leverage: <StyledText className="font-bold">{item.resultSummary.leverage}x</StyledText></StyledText>
                                </StyledView>
                            </StyledView>
                        ))}
                    </StyledScrollView>
                )}
            </StyledView>
        </SafeAreaView>
    );
}
