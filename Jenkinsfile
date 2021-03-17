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
        buildDiscarder(logRotator(numToKeepStr: '5'))
    }

    parameters {
        string(name: 'FBX_SDK_DIR', defaultValue: '', description: 'Path to the FBX SDK installation')
        string(name: 'OPTIX_DIR', defaultValue: '', description: 'Path to the Optix SDK installation')

        booleanParam(name: 'SOFTBODY_TESTBED_QT', defaultValue: true, description: 'Should testbed_qt be built (requires Qt5)')
        booleanParam(name: 'SOFTBODY_ENABLE_CUDA', defaultValue: true, description: 'Should the CUDA backend in softbody be built')
        booleanParam(name: 'SOFTBODY_ENABLE_TRACY', defaultValue: true, description: 'Should softbody be built with instrumentation')
        booleanParam(name: 'SOFTBODY_CLANG_TIDY', defaultValue: true, description: 'Should we run clang-tidy')
        booleanParam(name: 'CMAKE_EXPORT_COMPILE_COMMANDS', defaultValue: true, description: 'Should compile commands be exported')

        string(name: 'CMAKE_CC_COMPILER', defaultValue: 'clang', description: 'Path to the C compiler')
        string(name: 'CMAKE_CXX_COMPILER', defaultValue: 'clang++', description: 'Path to the C++ compiler')
        string(name: 'CMAKE_CUDA_COMPILER', defaultValue: '/usr/local/cuda/bin/nvcc', description: 'Path to the CUDA compiler (nvcc)')
        string(name: 'CLANG_TIDY', defaultValue: 'clang-tidy', description: 'Path to clang-tidy')
    }

    stages {
        stage('Mark build as pending') {
            steps {
                setBuildStatus("Build has started", "PENDING");
            }
        }

        stage('Configure') {
            steps {
                cmake arguments: '-DFBX_SDK_DIR=${params.FBX_SDK_DIR} -DOPTIX_DIR=${params.OPTIX_DIR} -DSOFTBODY_TESTBED_QT=${params.SOFTBODY_TESTBED_QT} -DSOFTBODY_ENABLE_CUDA=${params.SOFTBODY_ENABLE_CUDA} -DSOFTBODY_ENABLE_TRACY=${params.SOFTBODY_ENABLE_TRACY} -DCMAKE_EXPORT_COMPILE_COMMANDS=${params.CMAKE_EXPORT_COMPILE_COMMANDS} -DCMAKE_CC_COMPILER=${params.CMAKE_CC_COMPILER} -DCMAKE_CXX_COMPILER=${params.CMAKE_CXX_COMPILER} -DCMAKE_CUDA_COMPILER=${params.CMAKE_CUDA_COMPILER} -DSOFTBODY_CLANG_TIDY=${params.SOFTBODY_CLANG_TIDY} -DCLANG_TIDY=${params.CLANG_TIDY}', installation: 'InSearchPath'
            }
        }
        stage('Build') {
            steps {
                cmakeBuild buildType: 'Release', cleanBuild: true, installation: 'InSearchPath', steps: [[withCmake: true]]
            }
        }
        stage('Record clang warnings') {
            steps {
                recordIssues(tools: [clang())])
            }
        }
        stage('Record clang-tidy issues') {
            when {
                expression { 
                    return params.SOFTBODY_CLANG_TIDY
                }
            }
            steps {
                recordIssues(tools: [clangTidy()])
            }
        }
    }
    post {
        success {
            setBuildStatus("Build has started", "SUCCESS");
        }
        failure {
            setBuildStatus("Build has started", "FAILURE");
        }
    }
}
