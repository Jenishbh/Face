import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import UploadScreen from '../home/home';
import TrimResultsScreen from '../home/output';

const Stack = createNativeStackNavigator();

const App = () => {
  return (
    <NavigationContainer independent={true}>
      <Stack.Navigator initialRouteName="Upload">
        <Stack.Screen name="Upload" component={UploadScreen} />
        <Stack.Screen name="TrimResults" component={TrimResultsScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;
