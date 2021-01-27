import nvtabular
import cudf
from tritonclient.utils import np_to_triton_dtype
import tritonclient.http as httpclient
import timeit


def triton_infer(batch, workflow, model, host="localhost:8000"):
    # convert the batch to a triton inputs
    columns = [(col, batch[col]) for col in workflow.column_group.input_column_names]
    inputs = [
        httpclient.InferInput(name, col.shape, np_to_triton_dtype(col.dtype))
        for name, col in columns
    ]
    for i, (name, col) in enumerate(columns):
        inputs[i].set_data_from_numpy(col.values_host)

    # placeholder variables for the output
    outputs = [httpclient.InferRequestedOutput(name) for name in workflow.column_group.columns]

    # make the request
    with httpclient.InferenceServerClient(host) as client:
        response = client.infer(model, inputs, request_id="1", outputs=outputs)

    return cudf.DataFrame({col: response.as_numpy(col) for col in workflow.column_group.columns})


def local_infer(batch, workflow):
    return nvtabular.workflow._transform_partition(batch, [workflow.column_group])


def benchmark_model(model, trials=3):
    workflow = nvtabular.Workflow.load("./" + model)
    train = cudf.read_csv("./data/train.csv")[workflow.column_group.input_column_names]

    with open(f"{model}.csv", "w") as o:
        o.write("batchsize, local (ms), triton (ms)\n")
        for i in range(17):
            batch_size = 2 ** i
            batch = train.head(batch_size)

            local_time = min(
                timeit.Timer(lambda: local_infer(batch, workflow)).repeat(repeat=trials, number=1)
            )
            triton_time = min(
                timeit.Timer(lambda: triton_infer(batch, workflow, model)).repeat(
                    repeat=trials, number=1
                )
            )

            local_time *= 1000
            triton_time *= 1000

            print(f"{batch_size}, {local_time}, {triton_time}")
            o.write(f"{batch_size}, {local_time}, {triton_time}\n")


if __name__ == "__main__":
    benchmark_model("rossmann_categorify")
    benchmark_model("rossmann_hashbucket")
