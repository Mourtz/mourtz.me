// Include standard headers
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdlib.h>
#include <sstream>
#include <string.h>
// Include GLEW
#include <GL/glew.h>
// Include GLFW
#include <glfw3.h>
GLFWwindow* window;
// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
// https://github.com/nlohmann/json
#include <json.hpp>

using namespace std;
using namespace glm;
using json = nlohmann::json;

GLfloat iterations = 5;
// window size in pixels
int window_width = 1024, window_height = 768;
float aspect_ratio = window_width / window_height;
float fov = 45.0f;
float zfar = 1024.0f, znear = 0.1f;

mat4 CreateViewMatrix(vec3 position, vec3 direction, vec3 up) {
	vec3 f = direction;
	float len = sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
	f = vec3(f[0] / len, f[1] / len, f[2] / len);

	vec3 s = vec3(up[1] * f[2] - up[2] * f[1],
		up[2] * f[0] - up[0] * f[2],
		up[0] * f[1] - up[1] * f[0]);

	len = sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);

	vec3 s_norm = vec3(s[0] / len,
		s[1] / len,
		s[2] / len);

	vec3 u = vec3(f[1] * s_norm[2] - f[2] * s_norm[1],
		f[2] * s_norm[0] - f[0] * s_norm[2],
		f[0] * s_norm[1] - f[1] * s_norm[0]);

	vec3 p = vec3(-position[0] * s_norm[0] - position[1] * s_norm[1] - position[2] * s_norm[2],
		-position[0] * u[0] - position[1] * u[1] - position[2] * u[2],
		-position[0] * f[0] - position[1] * f[1] - position[2] * f[2]);

	return mat4(s_norm[0], u[0], f[0], 0.0,
		s_norm[1], u[1], f[1], 0.0,
		s_norm[2], u[2], f[2], 0.0,
		p[0], p[1], p[2], 1.0);
}

mat4 CreatePerspectiveMatrix() {
	float f = tan(3.141592f * 0.5 - 0.5 * fov);
	float range = 1.0 / (znear - zfar);
	return mat4(
		f / aspect_ratio, 0.0, 0.0, 0.0,
		0.0, f, 0.0, 0.0,
		0.0, 0.0, (znear + zfar) * range, -1.0,
		0.0, 0.0, znear * zfar * range * 2, 0.0
	);
}

string readFile(const char *filePath) {
	string content;
	ifstream fileStream(filePath, ios::in);

	if (!fileStream.is_open()) {
		cerr << "Could not read file " << filePath << ". File does not exist." << endl;
		return "";
	}

	string line = "";
	while (!fileStream.eof()) {
		getline(fileStream, line);
		content.append(line + "\n");
	}

	fileStream.close();
	return content;
}

static mat4 viewMatrix = CreateViewMatrix(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0));

//static mat4 perspectiveMatrix = perspective(fov, aspect_ratio, znear, zfar);
static mat4 perspectiveMatrix = CreatePerspectiveMatrix();

// on window resize callback
void windowSizeCallback(GLFWwindow* window, int width, int height)
{
	window_width = width;
	window_height = height;
	aspect_ratio = window_width / window_height;
	glViewport(0, 0, width, height);
}

int main(void)
{
	// Initialise GLFW
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		getchar();
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

	// Open a window and create its OpenGL context
	window = glfwCreateWindow(window_width, window_height, "GL Viewport", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window.\n");
		getchar();
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return -1;
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	// Black background
	glClearColor(0.0, 0.0, 0.0, 0.0);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);
	// Cull triangles which normal is not towards the camera
	glEnable(GL_CULL_FACE);

	// Create and compile our GLSL program from the shaders
	// GLuint programID = LoadShaders("SimpleVertexShader.vertexshader", "SimpleFragmentShader.fragmentshader");
	string VertexShaderCode = readFile("SimpleVertexShader.vertexshader");

	string FragmentShaderCode = readFile("SimpleFragmentShader.fragmentshader");

	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	GLint Result = GL_FALSE;
	int InfoLogLength;

	// compile vertex shader
	char const * VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource(VertexShaderID, 1, &VertexSourcePointer, NULL);
	glCompileShader(VertexShaderID);

	// check vertex shader
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		printf("%s\n", &VertexShaderErrorMessage[0]);
	}

	// compile fragment shader
	char const * FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, NULL);
	glCompileShader(FragmentShaderID);

	// check fragment shader
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
		printf("%s\n", &FragmentShaderErrorMessage[0]);
	}

	GLuint programID = glCreateProgram();
	glAttachShader(programID, VertexShaderID);
	glAttachShader(programID, FragmentShaderID);
	glLinkProgram(programID);

	// check the program
	glGetProgramiv(programID, GL_LINK_STATUS, &Result);
	glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
		glGetProgramInfoLog(programID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		printf("%s\n", &ProgramErrorMessage[0]);
	}


	// Uniform Locations
	GLuint perspectiveMatrixID = glGetUniformLocation(programID, "perspective");
	GLuint modelMatrixID = glGetUniformLocation(programID, "model");
	GLuint viewMatrixID = glGetUniformLocation(programID, "view");
	GLuint timeID = glGetUniformLocation(programID, "u_time");
	GLuint resolutionID = glGetUniformLocation(programID, "u_resolution");

	cout << "Loading suzanne_high-res.json..." << endl;
	ifstream file("suzanne_high-res.json");
	stringstream buffer;
	buffer << file.rdbuf();
	json js = json::parse(buffer.str());
	cout << "vertices: " << js["vertexPositions"].size() << endl;
	cout << "normals: " << js["vertexNormals"].size() << endl;
	cout << "indices: " << js["indices"].size() << endl;

	vector<GLfloat> vertices;
	vector<GLuint> indices;
	vector<GLfloat> normals;

	vertices.insert(vertices.end(), &js["vertexPositions"][0], &js["vertexPositions"][js["vertexPositions"].size()]);
	indices.insert(indices.end(), &js["indices"][0], &js["indices"][js["indices"].size()]);
	normals.insert(normals.end(), &js["vertexNormals"][0], &js["vertexNormals"][js["vertexNormals"].size()]);

	//loadOBJ("suzanne_high-res.obj", vertices, uvs, normals);

	//VBOs
	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat), &vertices[0], GL_STATIC_DRAW);

	GLuint normalbuffer;
	glGenBuffers(1, &normalbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
	glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(GLfloat), &normals[0], GL_STATIC_DRAW);

	GLuint indicesbuffer;
	glGenBuffers(1, &indicesbuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicesbuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), &indices[0], GL_STATIC_DRAW);


	// Get a handle for our buffers
	GLuint a_PostionID = glGetAttribLocation(programID, "position");
	GLuint a_NormalID = glGetAttribLocation(programID, "normal");

	// 1st attribute buffer : vertices
	glEnableVertexAttribArray(a_PostionID);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(
		a_PostionID,        // attribute 0. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);

	// 2nd attribute buffer : normals
	glEnableVertexAttribArray(a_NormalID);
	glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
	glVertexAttribPointer(
		a_NormalID,                       // attribute
		3,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		0,                                // stride
		(void*)0                          // array buffer offset
	);

	// use loaded program
	glUseProgram(programID);
	// set on resize callback function
	glfwSetWindowSizeCallback(window, windowSizeCallback);

	double lastTime = glfwGetTime();
	int nbFrames = 0;

	do {

		// Measure speed
		GLfloat currentTime = glfwGetTime();
		nbFrames++;
		if (currentTime - lastTime >= 1.0) { // If last prinf() was more than 1 sec ago
											 // printf and reset timer
			printf("%f ms/frame\n", 1000.0 / double(nbFrames));
			nbFrames = 0;
			lastTime += 1.0;
		}

		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		for (GLfloat x = 0; x < iterations; x++) {
			GLfloat size = 60 / iterations * 0.01;
			GLfloat _step = iterations / 2;
			GLfloat posX = -0.9 + x / _step;
			GLfloat posY = 1;
			GLfloat posZ = -2;

			for (GLfloat y = 0; y < iterations; y++) {
				posY = 0.9 - y / _step;

				for (GLfloat z = 0; z < iterations; z++) {
					posZ = -2 - z / _step;

					//  Window Resolution Uniform
					glUniform2f(resolutionID, (float)window_width, (float)window_height);
					// Time Uniform
					glUniform1f(timeID, currentTime);
					// Perspective Matrix Uniform
					glUniformMatrix4fv(perspectiveMatrixID, 1, GL_FALSE, &perspectiveMatrix[0][0]);
					// View Matrix Uniform
					glUniformMatrix4fv(viewMatrixID, 1, GL_FALSE, &viewMatrix[0][0]);
					// Model Matrix Uniform
					glUniformMatrix4fv(modelMatrixID, 1, GL_FALSE,
						new GLfloat[16]{
						size, 0.0, 0.0, 0.0,
						0.0, size, 0.0, 0.0,
						0.0, 0.0, size, 0.0,
						posX, posY, posZ, 1.0
					}
					);

					// Draw suzanne !
					//glDrawArrays(GL_TRIANGLES, 0, vertices.size());
					glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, (void*)0);
				}
			}

		}

		//viewMatrix[3][2] += 0.01;

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();
	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
		glfwWindowShouldClose(window) == 0);

	// Cleanup VBO
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteBuffers(1, &normalbuffer);
	// Delete program
	glDeleteProgram(programID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();

	return 0;
}
