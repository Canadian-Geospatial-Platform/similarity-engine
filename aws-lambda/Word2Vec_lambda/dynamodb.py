import boto3 
from botocore.exceptions import ClientError
import datetime
import os 

# See boto3 dynamoDB documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/dynamodb.html


#region = os.environ['REGION_NAME']
region = 'ca-central-1'
# Create a dynamodb table
def create_table_similarity(TableName, dynamodb=None):
    """     
    The size of the 'similarity' column contains large JSON objects, which exceeds the 1024 bytes limit of the DynamoDB key size.
    Non-key attributes in DynamoDB can be up to 400 KB in size, therefore, we will remove the 'similarity' column from the key schema.
    The primariary key will be the 'features_properties_id' column only. This might cause significant impact on the performance and cost-effectiveness of DynamoDB
    """
    dynamodb = boto3.resource('dynamodb', region_name=region)
    
    table = dynamodb.create_table(
        TableName=TableName,
        KeySchema=[
            {
                'AttributeName': 'features_properties_id',
                'KeyType': 'HASH'  # Partition key
            }
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'features_properties_id',
                'AttributeType': 'S'
            }
        ],
        BillingMode='PAY_PER_REQUEST'
    )
    # Wait until the table exists.
    table.meta.client.get_waiter('table_exists').wait(TableName=TableName)
    print(f"Table {TableName} created successfully!")
    return table
    
    
# Batch writing items into a table 
def batch_write_items_into_table(df, TableName):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(TableName)

    #Get current date and time
    dateTime = datetime.datetime.utcnow().now()
    dateTime = dateTime.isoformat()[:-7] + 'Z'
    
    with table.batch_writer() as batch:
        for i in range(len(df)):
            try: 
                batch.put_item(
                    Item={
                        'features_properties_id': df.loc[i, 'features_properties_id'],
                        'similarity': df.loc[i, 'similarity'],
                        'created_at': dateTime
                    }
                )
                
            except ClientError as e:
                print(e.response['Error']['Message'])
    print("All items added to the table successfully!")


#Delete a table  
def delete_table(TableName):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(TableName)
    # Check if the table exists
    try:
        response = table.delete()
        print(f"Table {TableName} deleted successfully!")
    except Exception as e:
        print("Error deleting table. It might not exist. Details:", e)