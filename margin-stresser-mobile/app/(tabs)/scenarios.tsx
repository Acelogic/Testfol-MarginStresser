import React, { useState } from 'react';
import { View, Text, TouchableOpacity, ScrollView, Alert, SafeAreaView } from 'react-native';
import { styled } from 'nativewind';
import { useRouter } from 'expo-router';
import { useAppContext } from '../context/AppContext';
import { IconSymbol } from '@/components/ui/icon-symbol';

const StyledView = styled(View);
const StyledText = styled(Text);
const StyledScrollView = styled(ScrollView);
const StyledTouchableOpacity = styled(TouchableOpacity);

export default function ScenariosScreen() {
    const { scenarios, deleteScenario, loadScenario } = useAppContext();
    const router = useRouter();

    const handleLoad = (scenario: any) => {
        loadScenario(scenario);
        router.push('/(tabs)');
        Alert.alert("Loaded", `Scenario "${scenario.name}" loaded.`);
    };

    const handleDelete = (id: string, name: string) => {
        Alert.alert(
            "Delete Scenario",
            `Are you sure you want to delete "${name}"?`,
            [
                { text: "Cancel", style: "cancel" },
                { text: "Delete", style: "destructive", onPress: () => deleteScenario(id) }
            ]
        );
    };

    return (
        <SafeAreaView className="flex-1 bg-gray-100">
            <StyledView className="flex-1 p-4">
                <StyledText className="text-3xl font-bold text-center mb-6 mt-2 text-gray-800">Saved Scenarios</StyledText>

                {scenarios.length === 0 ? (
                    <StyledView className="flex-1 justify-center items-center">
                        <StyledText className="text-gray-500 text-lg mb-2">No saved scenarios yet.</StyledText>
                        <StyledText className="text-gray-400 text-center px-6">
                            Go to the Home tab, set up a portfolio, and click "Save Scenario" to add one here.
                        </StyledText>
                    </StyledView>
                ) : (
                    <StyledScrollView className="flex-1" showsVerticalScrollIndicator={false}>
                        {scenarios.map((scenario) => (
                            <StyledView key={scenario.id} className="bg-white rounded-xl shadow-sm p-4 mb-4 border border-gray-200">
                                <StyledView className="flex-row justify-between items-center mb-2">
                                    <StyledText className="text-xl font-bold text-gray-800">{scenario.name}</StyledText>
                                    <StyledTouchableOpacity onPress={() => handleDelete(scenario.id, scenario.name)} className="p-2">
                                        <IconSymbol size={20} name="trash.fill" color="#ef4444" />
                                    </StyledTouchableOpacity>
                                </StyledView>

                                <StyledText className="text-gray-600 mb-2">
                                    Value: <StyledText className="font-bold">${parseFloat(scenario.portfolioValue).toLocaleString()}</StyledText> •
                                    Debt: <StyledText className="font-bold text-red-600">${parseFloat(scenario.marginDebt).toLocaleString()}</StyledText>
                                </StyledText>

                                <StyledView className="flex-row flex-wrap mb-3">
                                    {scenario.tickers.slice(0, 5).map((t, i) => (
                                        <StyledView key={i} className="bg-gray-100 rounded-md px-2 py-1 mr-2 mb-2">
                                            <StyledText className="text-xs font-medium text-gray-700">{t.symbol} ({t.percent}%)</StyledText>
                                        </StyledView>
                                    ))}
                                    {scenario.tickers.length > 5 && (
                                        <StyledText className="text-xs text-gray-500 mt-1">+{scenario.tickers.length - 5} more</StyledText>
                                    )}
                                </StyledView>

                                <StyledTouchableOpacity
                                    onPress={() => handleLoad(scenario)}
                                    className="bg-blue-600 p-3 rounded-lg flex-row justify-center items-center"
                                >
                                    <IconSymbol size={18} name="arrow.up.left.circle.fill" color="white" />
                                    <StyledText className="text-white font-bold ml-2">Load Scenario</StyledText>
                                </StyledTouchableOpacity>
                            </StyledView>
                        ))}
                    </StyledScrollView>
                )}
            </StyledView>
        </SafeAreaView>
    );
}
