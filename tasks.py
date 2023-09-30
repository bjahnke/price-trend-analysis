from src.util_invoke_tasks import *
import env


@task
def dockerbuild(c):
    """
    run docker daemon if not already running and build the docker image
    :return:
    """
    print("Building docker image...")
    # print current working directory
    image_name = get_env_var('IMAGE_NAME')
    subprocess.run(
        f"docker build --ssh github_ssh_key=./id_rsa "
        f"-t {image_name} -f docker/Dockerfile . --no-cache",
        shell=True
    )


@task
def gcrdeploy(c):
    """
    Deploy the docker image to Google Cloud Run.
    :return:
    """
    print("Deploying docker image to Google Cloud Run...")
    tag = 'latest'
    docker_username = get_env_var('DOCKER_USERNAME')
    image_name = get_env_var('IMAGE_NAME')
    docker_tag = f'docker.io/{docker_username}/{image_name}:{tag}'
    envtoyaml(c)
    command = [
        'gcloud',
        'run',
        'deploy',
        get_env_var('IMAGE_NAME'),
        '--image',
        docker_tag,
        '--region',
        'us-east1',
        '--no-allow-unauthenticated',
        '--project',
        get_env_var('GCR_PROJECT_ID'),
        '--env-vars-file',
        './env.yaml',
        '--memory',
        '4Gi',
        '--timeout',
        '30m'

    ]
    print(' '.join(command))
    subprocess.run(command, check=True, shell=True)
