import React, { useEffect, useState } from "react";
import { View, Image, TextInput, Text, StyleSheet } from "react-native";
import AsyncStorage from '@react-native-async-storage/async-storage';

function Profile() {
  const [user, setUser] = useState(null);
  const [userData, setUserData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    gender: '',
    dateOfBirth: '',
    phoneNumber: '',
  });

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const token = await AsyncStorage.getItem('token');

        const response = await fetch("http://192.168.72.6:3000/api/v1/user", {
          method: "GET",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        const data = await response.json();

        if (response.ok) {
          setUser(data.firstName + ' ' + data.lastName);
          setUserData({
            ...data,
            dateOfBirth: formatDate(data.dateOfBirth), // Format the dateOfBirth field
          });
        } else {
          console.error(data.message);
        }
      } catch (error) {
        console.error(error);
      }
    };

    fetchUserData();
  }, []);

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const formattedDate = `${date.getMonth() + 1}-${date.getDate()}-${date.getFullYear()}`;
    return formattedDate;
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Profile</Text>
      <View style={styles.profileSection}>
        <Image
          style={styles.profile}
          source={require("../assets/images/profile.jpg")}
        />
        <Text style={styles.profileName}>{user}</Text>
      </View>
      <View style={styles.inputContainer}>
        <TextInput
          placeholder={userData.firstName}
          placeholderTextColor="black"
          style={styles.input}
          editable={false}
        />
        <TextInput
          placeholder={userData.lastName}
          placeholderTextColor="black"
          style={styles.input}
          editable={false}
        />
        <TextInput
          placeholder={userData.email}
          placeholderTextColor="black"
          style={styles.input}
          editable={false}
        />
        <TextInput
          placeholder={userData.gender}
          placeholderTextColor="black"
          style={styles.input}
          editable={false}
        />
        <TextInput
          placeholder={userData.dateOfBirth}
          placeholderTextColor="black"
          style={styles.input}
          editable={false}
        />
        <TextInput
          placeholder={userData.phoneNumber}
          placeholderTextColor="black"
          style={styles.input}
          editable={false}
        />
      </View>
    </View>
  );
}

export default Profile;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
  },
  title: {
    fontWeight: "bold",
    fontSize: 24,
    backgroundColor: "#162544",
    color: "white",
    padding: 20,
    textAlign: "center",
  },
  profileSection: {
    backgroundColor: "#162544",
    alignItems: "center",
    paddingTop: 40,
    paddingBottom: 20,
  },
  profile: {
    height: 100,
    width: 100,
    borderRadius: 50,
    marginBottom: 10,
  },
  profileName: {
    color: "white",
    fontSize: 20,
    marginBottom: 10,
  },
  inputContainer: {
    paddingHorizontal: 20,
    marginTop: 20,
  },
  input: {
    height: 50,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#ccc",
    paddingHorizontal: 10,
    marginBottom: 10,
  },
});
