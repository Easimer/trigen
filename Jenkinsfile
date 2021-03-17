void setBuildStatus(String message, String state) {
  step([
      $class: "GitHubCommitStatusSetter",
      reposSource: [$class: "ManuallyEnteredRepositorySource", url: "https://github.com/Easimer/trigen"],
      contextSource: [$class: "ManuallyEnteredCommitContextSource", context: "ci/jenkins/build-status"],
      errorHandlers: [[$class: "ChangingBuildStatusErrorHandler", result: "UNSTABLE"]],
      statusResultSource: [ $class: "ConditionalStatusResultSource", results: [[$class: "AnyBuildResult", message: message, state: state]] ]
  ]);
}

pipeline {
    agent any

    options {
        buildDiscarder(logRotator(numToKeepStr: '30', artifactNumToKeepStr: '2'))
    }

    parameters {
        string(name: 'FBX_SDK_DIR', defaultValue: '/srv/libraries/fbxsdk', description: 'Path to the FBX SDK installation')
        string(name: 'OPTIX_DIR', defaultValue: '/srv/libraries/NVIDIA-OptiX-SDK-7.2.0-linux64-x86_64', description: 'Path to the Optix SDK installation')

        string(name: 'SOFTBODY_TESTBED_QT', defaultValue: "ON", description: 'Should testbed_qt be built (requires Qt5)')
        string(name: 'SOFTBODY_ENABLE_CUDA', defaultValue: "ON", description: 'Should the CUDA backend in softbody be built')
        string(name: 'SOFTBODY_ENABLE_TRACY', defaultValue: "ON", description: 'Should softbody be built with instrumentation')
        string(name: 'SOFTBODY_CLANG_TIDY', defaultValue: "ON", description: 'Should we run clang-tidy')
        string(name: 'CMAKE_EXPORT_COMPILE_COMMANDS', defaultValue: "ON", description: 'Should compile commands be exported')

        string(name: 'CMAKE_C_COMPILER', defaultValue: '/usr/bin/clang', description: 'Path to the C compiler')
        string(name: 'CMAKE_CXX_COMPILER', defaultValue: '/usr/bin/clang++', description: 'Path to the C++ compiler')
        string(name: 'CMAKE_CUDA_COMPILER', defaultValue: '/usr/local/cuda/bin/nvcc', description: 'Path to the CUDA compiler (nvcc)')
        string(name: 'CLANG_TIDY', defaultValue: '/usr/bin/clang-tidy', description: 'Path to clang-tidy')
    }

    stages {
        stage('Mark build as pending') {
            steps {
                setBuildStatus("Build has started", "PENDING");
            }
        }

        stage('Configure') {
            steps {
                cmakeBuild buildType: 'Release', cleanBuild: true, installation: 'InSearchPath', buildDir: 'build', cmakeArgs: "-DFBX_SDK_DIR=${params.FBX_SDK_DIR} -DOPTIX_DIR=${params.OPTIX_DIR} -DSOFTBODY_TESTBED_QT=${params.SOFTBODY_TESTBED_QT} -DSOFTBODY_ENABLE_CUDA=${params.SOFTBODY_ENABLE_CUDA} -DSOFTBODY_ENABLE_TRACY=${params.SOFTBODY_ENABLE_TRACY} -DCMAKE_EXPORT_COMPILE_COMMANDS=${params.CMAKE_EXPORT_COMPILE_COMMANDS} -DCMAKE_C_COMPILER=${params.CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${params.CMAKE_CXX_COMPILER} -DCMAKE_CUDA_COMPILER=${params.CMAKE_CUDA_COMPILER} -DSOFTBODY_CLANG_TIDY=${params.SOFTBODY_CLANG_TIDY} -DCLANG_TIDY=${params.CLANG_TIDY}"
            }
        }
        stage('Build') {
            steps {
                cmakeBuild buildDir: 'build', installation: 'InSearchPath', steps: [ [args: 'all'] ]
            }
        }
        stage('Record warnings') {
            parallel {
                stage('Record clang warnings') {
                    steps {
                        recordIssues(tools: [clang()])
                    }
                }
                stage('Record clang-tidy issues') {
                    when {
                        expression { 
                            return params.SOFTBODY_CLANG_TIDY == "ON"
                        }
                    }
                    steps {
                        recordIssues(tools: [clangTidy()])
                    }
                }
            }
        }
    }
    post {
        success {
            setBuildStatus("Build was successful", "SUCCESS");

            sshagent (credentials: ['git']) {
                sh('git checkout master')
                sh('git merge --ff-only develop')
                sh('git push origin master')
            }
        }
        failure {
            setBuildStatus("Build failed", "FAILURE");
        }
    }
}
