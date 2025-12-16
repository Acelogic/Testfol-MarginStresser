import React from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView, SafeAreaView, Alert } from 'react-native';
import { styled } from 'nativewind';
import { useAppContext } from '../context/AppContext';

const StyledView = styled(View);
const StyledText = styled(Text);
const StyledTextInput = styled(TextInput);
const StyledScrollView = styled(ScrollView);
const StyledTouchableOpacity = styled(TouchableOpacity);

export default function SettingsScreen() {
    const { settings, updateSettings } = useAppContext();

    // Local state for inputs to allow editing before saving (or just save on blur/change)
    // Accessing context directly for values.

    const handleSaveDefaults = (rate: string, margin: string) => {
        updateSettings({ defaultInterestRate: rate, defaultMaintenanceMargin: margin });
    };

    return (
        <SafeAreaView className="flex-1 bg-gray-100">
            <StyledView className="flex-1 p-4">
                <StyledText className="text-3xl font-bold text-center mb-6 mt-2 text-gray-800">Settings</StyledText>

                <StyledScrollView className="flex-1" showsVerticalScrollIndicator={false}>
                    {/* Defaults Section */}
                    <StyledView className="bg-white rounded-xl shadow-sm p-5 mb-5 border border-gray-200">
                        <StyledText className="text-xl font-semibold mb-4 text-gray-700">Default Simulation Values</StyledText>

                        <StyledView className="mb-4">
                            <StyledText className="text-gray-600 mb-1 font-medium">Default Interest Rate (%)</StyledText>
                            <StyledTextInput
                                value={settings.defaultInterestRate}
                                onChangeText={(v) => updateSettings({ defaultInterestRate: v })}
                                keyboardType="numeric"
                                className="border border-gray-300 rounded-lg p-3 bg-gray-50 text-gray-800"
                            />
                        </StyledView>

                        <StyledView className="mb-2">
                            <StyledText className="text-gray-600 mb-1 font-medium">Default Maintenance Margin (%)</StyledText>
                            <StyledTextInput
                                value={settings.defaultMaintenanceMargin}
                                onChangeText={(v) => updateSettings({ defaultMaintenanceMargin: v })}
                                keyboardType="numeric"
                                className="border border-gray-300 rounded-lg p-3 bg-gray-50 text-gray-800"
                            />
                        </StyledView>
                    </StyledView>

                    {/* Appearance Section */}
                    <StyledView className="bg-white rounded-xl shadow-sm p-5 mb-5 border border-gray-200">
                        <StyledText className="text-xl font-semibold mb-4 text-gray-700">Appearance</StyledText>
                        <StyledView className="flex-row bg-gray-100 p-1 rounded-lg">
                            {['light', 'dark', 'system'].map((mode) => {
                                const isActive = settings.theme === mode;
                                return (
                                    <StyledTouchableOpacity
                                        key={mode}
                                        onPress={() => updateSettings({ theme: mode as any })}
                                        className={`flex-1 p-2 rounded-md ${isActive ? 'bg-white shadow-sm' : ''}`}
                                    >
                                        <StyledText className={`text-center capitalize font-medium ${isActive ? 'text-blue-600' : 'text-gray-500'}`}>{mode}</StyledText>
                                    </StyledTouchableOpacity>
                                );
                            })}
                        </StyledView>
                        <StyledText className="text-gray-400 text-xs mt-2 italic">Note: Dark mode support is incomplete in this beta.</StyledText>
                    </StyledView>

                    {/* About */}
                    <StyledView className="items-center mt-4">
                        <StyledText className="text-gray-400 text-sm">Testfol Margin Stresser v1.0.0</StyledText>
                        <StyledText className="text-gray-300 text-xs">Powered by Testfol.io</StyledText>
                    </StyledView>

                </StyledScrollView>
            </StyledView>
        </SafeAreaView>
    );
}
