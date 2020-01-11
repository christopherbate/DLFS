#include "lib/GPU.hpp"
#include "lib/Logging.hpp"
#include <fstream>
#include <iostream>

#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include "proto/dlfs.grpc.pb.h"
#include "proto/dlfs.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;

class ServiceImpl final : public server::DLFSService::Service {
    grpc::Status StartTraining(grpc::ServerContext *context,
                               const server::TrainingRequest *request,
                               server::TrainingStatus *response) {
        std::string result("Started training");
        response->set_status(result);
        return Status::OK;
    }
};

int main() {
    std::string server_address("localhost:8585");
    ServiceImpl service;

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());

    std::cout << "Server listening on " << server_address << std::endl;

    server->Wait();
    return 0;
}