import React, { useEffect, useState } from "react";
import { Image, FlatList, StyleSheet, Text, View } from "react-native";
import { useNavigation } from "@react-navigation/native";
import SubjectsList from "../components/SubjectsList";
import AsyncStorage from "@react-native-async-storage/async-storage";
import SubjectDetails from "./SubjectDetails";

function HomePage() {
  const navigation = useNavigation();
  const [user, setUser] = useState(null);
  const [subjects, setSubjects] = useState([]);

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const token = await AsyncStorage.getItem("token");

        const response = await fetch("http://192.168.72.6:3000/api/v1/user", {
          method: "GET",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        const data = await response.json();

        if (response.ok) {
          setUser(data.firstName + " " + data.lastName);
          setSubjects(data.subjects);
          console.log(subjects);
        } else {
          console.error(data.message);
        }
      } catch (error) {
        console.error(error);
      }
    };

    fetchUserData();
  }, []);

  return (
    <View style={styles.container}>
      <View style={styles.profileSection}>
        <Image
          style={styles.profile}
          source={require("../assets/images/profile.jpg")}
        />
        <Text style={styles.profileName}>{user}</Text>
        <Text style={styles.profileLoc}>PCCE, Verna</Text>
      </View>
      <View style={styles.container2}>
        <Text style={styles.subjectsTitle}>Your Subjects</Text>
        <FlatList
          data={subjects}
          renderItem={({ item }) => (
            <SubjectsList
              text={item}
              id={item}
              onPress={() =>
                navigation.navigate("SubjectDetails", { id: item })
              }
            />
          )}
          keyExtractor={(item, index) => item}
          style={{ flex: 1 }}
        />
      </View>
    </View>
  );
}

export default HomePage;

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  profileSection: {
    backgroundColor: "#162544",
    height: 200,
    margin: 20,
    borderRadius: 25,
  },
  profile: {
    height: 80,
    width: 80,
    borderRadius: 40,
    marginTop: 25,
    margin: 20,
  },
  profileName: {
    color: "white",
    fontSize: 20,
    marginLeft: 15,
  },
  profileLoc: {
    color: "white",
    marginLeft: 18,
  },
  container2: {
    flex: 1,
    alignItems: "center",
  },
  subjectsTitle: {
    fontWeight: "bold",
    fontSize: 24,
  },
});
