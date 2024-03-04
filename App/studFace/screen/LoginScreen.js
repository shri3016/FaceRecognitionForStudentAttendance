import React, { useState } from "react";
import {
  Text,
  View,
  TextInput,
  StyleSheet,
  TouchableOpacity,
} from "react-native";
import AsyncStorage from '@react-native-async-storage/async-storage';

function LoginScreen({ navigation }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  async function handleLogin() {
    if (!email || !password) {
      alert('Enter the email and password');
      return;
    }

    try {
      const response = await fetch("http://192.168.72.6:3000/api/v1/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      })
      const data = await response.json();

      if (data.success) {
       
        await AsyncStorage.setItem('token', data.token);
        navigation.navigate('Home');
      } else {
        alert(data.message);
      }
    } catch (error) {
      console.error(error);
      alert("An error occurred during login. Please try again.");
    }

    setEmail("");
    setPassword("");
  }

  function handleForgotPassword() {
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Login</Text>
      <TextInput
        style={styles.input}
        value={email}
        placeholder="Email"
        autoCapitalize="none"
        onChangeText={setEmail}
      />
      <TextInput
        style={styles.input}
        value={password}
        placeholder="Password"
        autoCaptilize="none"
        onChangeText={setPassword}
        secureTextEntry={true}
      />
      {/* <TouchableOpacity
        style={styles.forgotPasswordText}
        onPress={handleForgotPassword}
      >
        <Text>forgot password?</Text>
      </TouchableOpacity> */}
      <TouchableOpacity style={styles.button} onPress={handleLogin}>
        <Text style={styles.buttonText}>Login</Text>
      </TouchableOpacity>
    </View>
  );
}

export default LoginScreen;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  title: {
    fontWeight: "bold",
    fontSize: 40,
    marginBottom: 30,
  },
  input: {
    width: "80%",
    height: 50,
    marginVertical: 10,
    padding: 10,
    borderWidth: 1,
    borderRadius: 5,
    borderColor: "#ccc",
  },
  forgotPasswordText: {
    color: "#162544",
    textDecorationLine: "underline",
    marginBottom: 20,
  },
  button: {
    width: "80%",
    height: 50,
    marginVertical: 20,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#162544",
    borderRadius: 5,
  },
  buttonText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "bold",
  },
});
