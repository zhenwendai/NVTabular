pushd ../../..
python setup.py bdist_wheel
popd
cp ../../../dist/*.whl nvtabular-0.3.0-py3-none-any.whl
docker build -t benfred/nvtabular:triton .
